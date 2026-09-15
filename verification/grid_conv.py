# -*- coding: utf-8 -*-
"""C2 의 잔여 -1%p 가 상태격자 이산화 몫인지 확인한다.

가설: 추적기는 0.0156 bin/심볼 의 램프를 0.03 격자 위에서 근사해야 하므로
평균이동 전이가 격자 사이로 질량을 퍼뜨린다 (스텝당 sig~step/2 의 확산이 주입됨).
격자를 촘촘히 하면 잔차가 같이 줄어야 한다.  안 줄면 구현 문제다.
"""
import os, io
import numpy as np
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
src = io.open(os.path.join(HERE, "genie_audit.py"), encoding="utf-8").read()
head = src.split("CONDS = [")[0]
g = {"__name__": "diag", "__file__": os.path.join(HERE, "genie_audit.py")}
exec(compile(head, "genie_audit_head", "exec"), g)

np_ = g["np"]; NS, N = g["NS"], g["N"]
ramp, rx, dec_drift = g["ramp"], g["rx"], g["dec_drift"]
lam_tensor, track, prior_eps0 = g["lam_tensor"], g["track"], g["prior_eps0"]
paired_boot = g["paired_boot"]; cluster_boot = g["cluster_boot"]
IRAMP = g["IRAMP"]

AMPV = 2.0
NPK = 12
STEPS = [0.06, 0.03, 0.015, 0.0075]
print(f"linear drift {AMPV} bin, {NPK} packets, 격자폭 +-1.5 고정, 간격만 바꾼다\n")

G = {}
for st in STEPS:
    n = int(round(1.5 / st))
    E = np.round(np.arange(-n, n + 1) * st, 8)
    G[st] = (E, np.stack([ramp(np.array([e]))[0] for e in E]), prior_eps0("linear", AMPV, E))

rng = np.random.default_rng(90210)
ok2d = np.zeros(NPK * NS, bool)
okt = {st: np.zeros(NPK * NS, bool) for st in STEPS}
rm = {st: [] for st in STEPS}
for k in range(NPK):
    s = rng.integers(0, N, NS)
    base = float(np.round(rng.uniform(-0.5, 0.5), 3))
    eps = base + AMPV * IRAMP
    Yp = rx(s, eps)
    sl = slice(k * NS, (k + 1) * NS)
    ok2d[sl] = dec_drift(Yp) == s
    for st in STEPS:
        E, V, pri = G[st]
        lam = lam_tensor(Yp, V)
        mh, eh = track(lam, E, st / 2, "exact", pri, AMPV / NS)
        okt[st][sl] = mh == s
        rm[st].append(float(np.sqrt(np.mean((eh - eps) ** 2))))
        del lam

m2, l2, h2, _ = cluster_boot(ok2d, NS, seed=7)
print(f"  2D drift-aware (참 모델, 연속 램프) = {m2:.1f}% [{l2:.1f},{h2:.1f}]\n")
print(f"  {'격자간격':>9}{'상태수':>7}{'추적기':>22}{'추적기 - 2D':>26}{'RMSE(eps)':>12}")
for st in STEPS:
    m, lo, hi, _ = cluster_boot(okt[st], NS, seed=7)
    d, dl, dh, p = paired_boot(okt[st], ok2d, NS, seed=7)
    print(f"  {st:>9.4f}{len(G[st][0]):>7}{m:>14.1f}% [{lo:4.1f},{hi:4.1f}]"
          f"{d:>14.2f}%p [{dl:5.2f},{dh:5.2f}]{np.mean(rm[st]):>12.4f}")
print("\n  간격을 줄일수록 잔차가 0 으로 수렴하면 이산화 몫이 맞다.")
