import numpy as np
import torch

from schnet import SchNet, ShiftedSoftplus, UniversalCutoff

torch.manual_seed(0)

data = np.load("data/rmd17_ethanol.npz")
z = torch.tensor(data["nuclear_charges"], dtype=torch.long)       # (9,)
pos = torch.tensor(data["coords"][:4], dtype=torch.float64)       # (4, 9, 3)

model = SchNet(F=64, act=ShiftedSoftplus(), cutoff_fn=UniversalCutoff()).double()

def energy_and_forces(p):
    p = p.clone().requires_grad_(True)
    E = model(z, p)
    F = -torch.autograd.grad(E.sum(), p)[0]
    return E.detach(), F.detach()

def random_rotation():
    A = torch.randn(3, 3, dtype=torch.float64) # 랜덤 행렬을 만들고 QR 분해한다.
    Q, _ = torch.linalg.qr(A)          # Q는 직교행렬
    if torch.det(Q) < 0:               # 행렬식 -1이면 반사이므로 첫째 열의 부호 뒤집어 고유 회전으로 만든다.
        Q[:, 0] = -Q[:, 0]
    return Q

E0, F0 = energy_and_forces(pos)

# 회전 + 평행이동 불변성 확인

R = random_rotation()
t = torch.randn(3, dtype=torch.float64) * 5.0

print("R is rotation:", torch.allclose(R @ R.T, torch.eye(3, dtype=torch.float64)),
      "det =", torch.det(R).item())

E1, F1 = energy_and_forces(pos @ R.T + t)

print("max |ΔE|        :", (E1 - E0).abs().max().item())          # ≈ 0 이어야 함
print("max |F1 - F0 Rᵀ|:", (F1 - F0 @ R.T).abs().max().item())    # ≈ 0 이어야 함

# auto-grad test

h = 1e-5
p0 = pos[:1]                                   # 샘플 하나 (1, 9, 3)
F_fd = torch.zeros_like(p0)

for i in range(9):          # 원자
    for k in range(3):      # x, y, z
        p_plus, p_minus = p0.clone(), p0.clone()
        p_plus[0, i, k] += h
        p_minus[0, i, k] -= h
        with torch.no_grad():
            F_fd[0, i, k] = -(model(z, p_plus) - model(z, p_minus)) / (2 * h)

print("max |F_autograd - F_fd|:", (F0[:1] - F_fd).abs().max().item())