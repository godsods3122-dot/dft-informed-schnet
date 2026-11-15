import math

import torch
import torch.nn.functional as F
from torch import nn

K_E = 332.0637  # 쿨롱 상수 (kcal/mol, A, e 단위)


# 거리, RBF, cutoff

def pairwise_distance(pos):  # pos: (B, N, 3)
    # None으로 축을 하나씩 끼워 넣어서 (B,N,1,3) - (B,1,N,3) 형태로 빼줌
    # 결과적으로 pos[b,i] - pos[b,j]가 계산됨
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    # i=j 이면 거리가 0이라 sqrt를 미분할 때 NaN이 나옴
    # 그래서 아주 작은 수(1e-9)를 더해줌
    return torch.sqrt((diff ** 2).sum(-1) + 1e-9)  # (B, N, N)


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf=20, cutoff=5.0):
        super().__init__()
        # 0부터 cutoff까지 가우시안 중심을 n_rbf개 균등하게 배치
        centers = torch.linspace(0, cutoff, n_rbf)
        self.register_buffer("centers", centers)  # 모델과 같이 GPU로 옮겨지도록 buffer로 등록
        self.gamma = 1.0 / (centers[1] - centers[0]).item() ** 2  # 폭은 중심 간격에 맞춤

    def forward(self, d):
        # 거리 하나당 가우시안 n_rbf개 값이 나옴: (..., n_rbf)
        return torch.exp(-self.gamma * (d[..., None] - self.centers) ** 2)


def cosine_cutoff(d, cutoff=5.0):
    # 원래 SchNet의 cutoff. cutoff 거리에서 0이 되고, 그 밖은 전부 0
    return 0.5 * (torch.cos(d * torch.pi / cutoff) + 1.0) * (d < cutoff)


class UniversalCutoff(nn.Module):
    # 이원자 DFT 전자밀도에서 fitting한 원소쌍별 cutoff
    #   r_star = R0 * (Z^q + Z'^q)          전자구름이 떨어지는 거리
    #   kappa  = k0/2 * (Z^-p + Z'^-p)      r_star 밖에서의 감쇠율
    #   f(r)   = cos(r) * exp(-kappa * (sp(r - r_star) - sp(0)))
    # sp는 softplus로 만든 매끄러운 max(0, x)
    # 기준점을 sp(0)으로 잡아서 r_star 안쪽에서는 cosine에 쌍마다 다른 상수가 곱해짐
    # (지금까지 돌린 실험과 같은 정의라 그대로 둠)

    def __init__(self, R0=1.7424, q=0.1000, k0=3.2790, p=0.3929, cutoff=5.0, width=0.2):
        super().__init__()
        self.R0, self.q, self.k0, self.p = R0, q, k0, p
        self.cutoff = cutoff
        self.beta = 1.0 / width  # softplus가 꺾이는 폭 (width가 작을수록 max에 가까움)

    def _sp(self, x):
        return F.softplus(x, beta=self.beta)

    def _pair_params(self, z):
        # 원자번호로부터 모든 원자쌍의 r_star와 kappa를 (N, N) 행렬로 만듦
        zf = z.float()
        Ra = self.R0 * zf ** self.q
        r_star = Ra[:, None] + Ra[None, :]       # 반지름처럼 더함
        ko = self.k0 * zf ** (-self.p)
        kappa = 0.5 * (ko[:, None] + ko[None, :])  # 감쇠율은 평균
        return r_star, kappa

    def forward(self, z, d):  # z: (N,), d: (B, N, N)
        r_star, kappa = self._pair_params(z)
        sp0 = self._sp(torch.zeros((), device=d.device, dtype=d.dtype))
        expo = -kappa * (self._sp(d - r_star) - sp0)
        return cosine_cutoff(d, self.cutoff) * torch.exp(expo)

    def outer_factor(self, z, d):
        # 정전기 스위치에 쓰는 겹침 지표 g(r)
        # r_star 안쪽에서는 거의 1이고, 밖에서는 kappa로 줄어듦
        # 여기서는 기준점을 sp(-r_star)로 잡아서 안쪽이 정확히 1 근처가 되게 함
        r_star, kappa = self._pair_params(z)
        return torch.exp(-kappa * (self._sp(d - r_star) - self._sp(-r_star)))


# 정전기 항 (PhysNet 참고)

def poly_cutoff(x):
    # PhysNet에서 쓰는 다항식 cutoff. x=1에서 0이 되고 2차 미분까지 연속
    x = x.clamp(max=1.0)
    return 1 - 6 * x ** 5 + 15 * x ** 4 - 10 * x ** 3


def shielded_coulomb(d):
    # PhysNet 방식의 쿨롱 거리 인자
    # 가까운 거리에서는 1/r이 발산하지 않게 완만하게 만들고, 5 A부터는 그냥 1/r
    N = d.shape[-1]
    eye = torch.eye(N, device=d.device, dtype=d.dtype)
    r = d + eye  # 대각선(자기 자신)에서 0으로 나누지 않도록 1을 더해둠
    phi = poly_cutoff(r / 5.0)
    chi = phi / torch.sqrt(r ** 2 + 1) + (1 - phi) / r
    return chi * (1 - eye)  # 자기 자신과의 상호작용은 빼기


# SchNet 구현

class Interaction(nn.Module):
    def __init__(self, F, n_rbf, act):
        super().__init__()
        self.filter_net = nn.Sequential(nn.Linear(n_rbf, F), act, nn.Linear(F, F))  # 거리로부터 필터 생성
        self.lin_in = nn.Linear(F, F)
        self.lin_out = nn.Sequential(nn.Linear(F, F), act, nn.Linear(F, F))

    def forward(self, x, rbf, fcut):
        W = self.filter_net(rbf)  # (B, N, N, F)
        W = W * fcut[..., None]   # cutoff를 곱해서 먼 원자의 영향을 줄임
        N = W.shape[1]
        # 항등행렬을 이용해 i=j(자기 자신)는 0으로 만듦
        W = W * (1 - torch.eye(N, device=W.device, dtype=W.dtype))[None, :, :, None]
        h = self.lin_in(x)        # (B, N, F)
        # cfconv: 원자 i에 대해 sum_j h_j * W_ij
        # h를 (B, 1, N, F)로 만들어 i 방향으로 복사한 뒤 j 방향으로 합침
        m = (h[:, None, :, :] * W).sum(dim=2)
        return x + self.lin_out(m)  # residual 연결


class SchNet(nn.Module):
    def __init__(self, F=64, n_rbf=20, cutoff=5.0, n_interaction=3, act=None, max_z=100,
                 cutoff_fn=None, elec=False, elec_mode="physnet"):
        super().__init__()
        self.embedding = nn.Embedding(max_z, F)  # 원자번호를 F차원 벡터로
        self.rbf = GaussianRBF(n_rbf, cutoff)
        self.cutoff = cutoff
        self.cutoff_fn = cutoff_fn  # None이면 cosine cutoff 사용
        self.interactions = nn.ModuleList([Interaction(F, n_rbf, act) for _ in range(n_interaction)])
        self.output = nn.Sequential(nn.Linear(F, F // 2), act, nn.Linear(F // 2, 1))  # 원자별 에너지

        self.elec = elec
        self.elec_mode = elec_mode
        self.last_q = None
        if elec:
            self.charge_head = nn.Linear(F, 1)  # 원자 특징에서 전하 하나를 뽑음
            # 0으로 초기화하면 안 됨
            # 정전기 에너지가 q의 2차식이라 q=0이면 기울기도 0이 되어 학습이 아예 안 됨
            nn.init.normal_(self.charge_head.weight, std=0.01)
            nn.init.zeros_(self.charge_head.bias)  # 어차피 평균을 빼면서 사라지는 값
            if elec_mode == "dft":
                self.switch = UniversalCutoff()

    def forward(self, z, pos):  # z: (N,) 원자번호, pos: (B, N, 3) 좌표
        B = pos.shape[0]
        d = pairwise_distance(pos)
        rbf = self.rbf(d)

        # 모든 분자가 원자 구성이 같으므로 임베딩을 B개로 복사
        x = self.embedding(z)[None].expand(B, -1, -1)  # (B, N, F)
        fcut = cosine_cutoff(d, self.cutoff) if self.cutoff_fn is None else self.cutoff_fn(z, d)

        for interaction in self.interactions:
            x = interaction(x, rbf, fcut)

        E = self.output(x).sum(dim=(1, 2))  # 원자별 에너지를 다 더해서 분자 에너지 (B,)

        if self.elec:
            q = self.charge_head(x).squeeze(-1)    # (B, N)
            q = q - q.mean(dim=-1, keepdim=True)   # 분자 전체 전하가 0이 되도록 평균을 뺌
            if self.elec_mode == "dft":
                # 전자구름이 겹치는 거리(r < r_star)에서는 정전기를 끄고 밖에서만 1/r
                N = d.shape[-1]
                eye = torch.eye(N, device=d.device, dtype=d.dtype)
                g = self.switch.outer_factor(z, d)
                chi = (1 - g) / (d + eye) * (1 - eye)
            else:
                chi = shielded_coulomb(d)
            # i, j 쌍을 양쪽으로 두 번 더하게 되므로 0.5를 곱함
            E = E + 0.5 * K_E * (q[:, :, None] * q[:, None, :] * chi).sum(dim=(1, 2))
            self.last_q = q.detach()  # 나중에 전하 값을 확인하려고 저장
        return E


class ShiftedSoftplus(nn.Module):
    def forward(self, x):
        return F.softplus(x) - math.log(2)  # ssp(0) = 0 이 되도록 log2를 뺌