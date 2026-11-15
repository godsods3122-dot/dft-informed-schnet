import argparse
import copy
import csv
import os
import time

import numpy as np
import torch
from torch import nn

from schnet import SchNet, ShiftedSoftplus, UniversalCutoff

# 1. 파라미터 설정

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/rmd17_ethanol.npz")
parser.add_argument("--width", type=int, default=64)                 # F
parser.add_argument("--n_interactions", type=int, default=3)         # T
parser.add_argument("--act", type=str, default="ssp", choices=["ssp", "gelu", "relu", "silu", "elu", "mish"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--patience", type=int, default=100)            # 10k run 은 500 으로 지정할 것
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--cutoff_mode", type=str, default="cosine", choices=["cosine", "universal"])
parser.add_argument("--elec", action="store_true")                   # 정전기 항 추가
parser.add_argument("--elec_mode", type=str, default="physnet", choices=["physnet", "dft"])
parser.add_argument("--out", type=str, default="results/results.csv")
args = parser.parse_args()

cutoff_fn = UniversalCutoff() if args.cutoff_mode == "universal" else None

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
np.random.seed(args.seed)

mol = os.path.basename(args.data).replace("rmd17_", "").replace(".npz", "")
tag = (f"{mol}_{args.cutoff_mode}"
       + (f"_elec-{args.elec_mode}" if args.elec else "")
       + f"_F{args.width}_T{args.n_interactions}_{args.act}_s{args.seed}")
print(f"run: {tag}  device: {device}")

# 2. 데이터 로드

data = np.load(args.data)
train_idx = np.loadtxt("data/index_train_01.csv", dtype=int)        # 분할 인덱스 01을 사용
test_idx = np.loadtxt("data/index_test_01.csv", dtype=int)

# 에너지가 매우 크므로, float64 에서 학습 평균을 뺀 잔차만 float32로 활용함
E_np = data["energies"].astype(np.float64)
E_mean64 = E_np[train_idx].mean()

z = torch.tensor(data["nuclear_charges"], dtype=torch.long, device=device)
pos_all = torch.tensor(data["coords"], dtype=torch.float32, device=device)
E_all = torch.tensor(E_np - E_mean64, dtype=torch.float32, device=device)
F_all = torch.tensor(data["forces"], dtype=torch.float32, device=device)

# 검증 세트: 학습·테스트 밖에서 1000개, 모든 실행에서 동일 (시드 0 고정)
rest = np.setdiff1d(np.arange(len(E_all)), np.concatenate([train_idx, test_idx]))
val_idx = np.random.default_rng(0).choice(rest, 1000, replace=False)

train_idx = torch.tensor(train_idx, device=device)
val_idx = torch.tensor(val_idx, device=device)
test_idx = torch.tensor(test_idx, device=device)

# 모델 설정

ACTS = {"ssp": ShiftedSoftplus, "gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "elu": nn.ELU, "mish": nn.Mish}

model = SchNet(F=args.width, n_interaction=args.n_interactions, act=ACTS[args.act](),
               cutoff_fn=cutoff_fn, elec=args.elec, elec_mode=args.elec_mode).to(device)
n_params = sum(p.numel() for p in model.parameters())

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=args.patience // 4)

W_E, W_F = 0.01, 0.99      # 손실 가중치 => F를 높게 평가함

# 4. 예측 및 평가

def predict(idx, create_graph):
    pos = pos_all[idx].clone().requires_grad_(True)
    E = model(z, pos)                                      # 평균을 뺀 잔차 스케일
    Fp = -torch.autograd.grad(E.sum(), pos, create_graph=create_graph)[0]
    return E, Fp


def evaluate(idx, chunk=250):
    model.eval()
    e_err, f_err = [], []
    for i in range(0, len(idx), chunk):
        b = idx[i:i + chunk]
        E, Fp = predict(b, create_graph=False)
        e_err.append((E - E_all[b]).abs().detach())
        f_err.append((Fp - F_all[b]).abs().detach())
    return torch.cat(e_err).mean().item(), torch.cat(f_err).mean().item()


def force_mae_by_element(idx, chunk=250):
    # 원소별 힘 MAE (H, C, O)
    model.eval()
    errs = []
    for i in range(0, len(idx), chunk):
        b = idx[i:i + chunk]
        _, Fp = predict(b, create_graph=False)
        errs.append((Fp - F_all[b]).abs().mean(-1).detach())   # (B, N)
    err = torch.cat(errs)
    return {el: err[:, z == el].mean().item() for el in [1, 6, 8] if (z == el).any()}


# 5. 학습

best_val, best_state, best_epoch, wait = float("inf"), None, 0, 0
hist = []
start = time.time()

for epoch in range(1, args.epochs + 1):
    model.train()
    perm = train_idx[torch.randperm(len(train_idx), device=device)]

    for i in range(0, len(perm), args.batch):
        b = perm[i:i + args.batch]
        E, Fp = predict(b, create_graph=True)
        loss = W_E * ((E - E_all[b]) ** 2).mean() + W_F * ((Fp - F_all[b]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_e, val_f = evaluate(val_idx)
    val_score = W_E * val_e + W_F * val_f
    scheduler.step(val_score)
    hist.append((epoch, val_e, val_f, optimizer.param_groups[0]["lr"]))

    if val_score < best_val:
        best_val, best_epoch, wait = val_score, epoch, 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        wait += 1

    if epoch % 20 == 0 or epoch == 1:
        print(f"epoch {epoch:5d} | val E MAE {val_e:.3f} | val F MAE {val_f:.3f} "
              f"| lr {optimizer.param_groups[0]['lr']:.1e} | {time.time() - start:.0f}s")

    if wait >= args.patience:
        print(f"early stop at epoch {epoch}")
        break

# 6. 최종 평가

model.load_state_dict(best_state)
tr_e, tr_f = evaluate(train_idx)
va_e, va_f = evaluate(val_idx)
te_e, te_f = evaluate(test_idx)
el_f = force_mae_by_element(test_idx)
elapsed = time.time() - start

print(f"\n[{tag}] params={n_params}  best_epoch={best_epoch}")
print(f"train E {tr_e:.3f}  F {tr_f:.3f}")
print(f"val   E {va_e:.3f}  F {va_f:.3f}")
print(f"test  E {te_e:.3f}  F {te_f:.3f}   (kcal/mol, kcal/mol/Å)")
names = {1: "H", 6: "C", 8: "O"}
print("test F by element: " + "  ".join(f"{names[k]} {v:.3f}" for k, v in el_f.items()))

mean_q = {}
if args.elec:
    predict(test_idx[:250], create_graph=False)
    q = model.last_q.mean(0)
    for el in [1, 6, 8]:
        if (z == el).any():
            mean_q[el] = q[z == el].mean().item()
    print("mean q: " + "  ".join(f"{names[k]} {v:+.3f} e" for k, v in mean_q.items()))

# 7. 결과 저장

path = args.out
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
row = dict(mol=mol, cutoff_fn=args.cutoff_mode,
           elec=int(args.elec), elec_mode=args.elec_mode if args.elec else "",
           width=args.width, depth=args.n_interactions, act=args.act, seed=args.seed,
           n_params=n_params, train_E=tr_e, train_F=tr_f, val_E=va_e, val_F=va_f,
           test_E=te_e, test_F=te_f,
           test_F_H=el_f.get(1, ""), test_F_C=el_f.get(6, ""), test_F_O=el_f.get(8, ""),
           q_H=mean_q.get(1, ""), q_C=mean_q.get(6, ""), q_O=mean_q.get(8, ""),
           best_epoch=best_epoch, epochs=args.epochs, patience=args.patience,
           time_s=round(elapsed))

if os.path.exists(path):                     # 헤더가 다르면 _v2 파일로
    with open(path) as f:
        if f.readline().strip().split(",") != list(row.keys()):
            path = path.replace(".csv", "_v2.csv")
new_file = not os.path.exists(path)
with open(path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    if new_file:
        writer.writeheader()
    writer.writerow(row)

os.makedirs("history", exist_ok=True)
np.savetxt(f"history/{tag}.csv", np.array(hist), delimiter=",",
           header="epoch,val_E,val_F,lr", comments="")
print(f"saved: {path}, history/{tag}.csv")