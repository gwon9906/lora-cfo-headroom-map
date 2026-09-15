# -*- coding: utf-8 -*-
"""[A2] genie 우위를 정수부(alias) 몫과 소수부 몫으로 분해한다.

심볼을 주변화한 우도는 eps 에 대해 주기 1 bin 으로 정확히 주기적이다
(CFO 1 bin 이동 == 데이터 심볼 1 칸 순환이동).  따라서 eps 의 정수부는 데이터
심볼과 교락되어 **심볼 단위로는 원리적으로 관측 불가능**하고, genie 는 그것을 안다.

  genie 소수부만 : eps - round(eps) 만 보상.  정수부는 그대로 남긴다.
  genie 정수부만 : round(eps) 만 보상.  남는 +-0.5 는 1D fine 추정기가 맡는다.

'정수부만' 이 genie 에 가깝고 '소수부만' 이 blind 에 가까우면, genie 우위의
대부분은 alias 정보이고 그것은 궤적 연속성으로만 회수 가능하다는 뜻이다.
"""
import os, io
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
src = io.open(os.path.join(HERE, "genie_audit.py"), encoding="utf-8").read()
head = src.split("CONDS = [")[0]
g = {"__name__": "diag", "__file__": os.path.join(HERE, "genie_audit.py")}
exec(compile(head, "genie_audit_head", "exec"), g)

NS, N, IRAMP = g["NS"], g["N"], g["IRAMP"]
ramp, rx = g["ramp"], g["rx"]
dec_blind, dec_const, dec_genie = g["dec_blind"], g["dec_const"], g["dec_genie"]
V_OFFF = g["V_OFFF"]
cluster_boot, paired_boot = g["cluster_boot"], g["paired_boot"]
NPK = 120

CONDS = [("walk", 0.0), ("walk", 0.5), ("walk", 1.0), ("walk", 2.0),
         ("linear", 1.0), ("linear", 2.0)]
ARMS = ["blind MF", "genie 소수부만", "genie 정수부만", "genie (전부)"]
print(f"SNR = -25 dB, ns = {NS}, {NPK} packets x {NS} = {NPK*NS} symbols/cell")
print("CI 는 packet-cluster bootstrap, 차이는 paired bootstrap.\n")
print(f"  {'조건':>13} |" + "".join(f"{a:>22}" for a in ARMS)
      + f"{'정수부 교락 몫':>24}{'정수부 넘나든 심볼':>20}")

rng = np.random.default_rng(90210)
for kind, amp in CONDS:
    ok = {a: np.zeros(NPK * NS, bool) for a in ARMS}
    nout = 0
    for k in range(NPK):
        s = rng.integers(0, N, NS)
        base = float(np.round(rng.uniform(-0.5, 0.5), 3))
        if kind == "linear":
            eps = base + amp * IRAMP
        else:
            w = np.cumsum(rng.normal(0, amp / np.sqrt(NS), NS)) if amp > 0 else np.zeros(NS)
            eps = base + w - w.mean()
        Yp = rx(s, eps)
        sl = slice(k * NS, (k + 1) * NS)
        nint = np.round(eps)
        nout += int((nint != 0).sum())
        ok["blind MF"][sl] = dec_blind(Yp) == s
        ok["genie 소수부만"][sl] = dec_genie(Yp, eps - nint) == s
        ok["genie 정수부만"][sl] = dec_const(Yp * ramp(nint), V_OFFF) == s
        ok["genie (전부)"][sl] = dec_genie(Yp, eps) == s
    row = f"  {kind+' '+format(amp,'.2f'):>13} |"
    for a in ARMS:
        m, lo, hi, _ = cluster_boot(ok[a], NS, seed=7)
        row += f"{m:>11.1f}% [{lo:4.1f},{hi:4.1f}]"
    d, dl, dh, p = paired_boot(ok["genie (전부)"], ok["genie 소수부만"], NS, seed=7)
    row += f"{d:>14.2f}%p [{dl:5.2f},{dh:5.2f}]{nout/(NPK*NS)*100:>19.1f}%"
    print(row)
print("\n  '정수부 교락 몫' = genie(전부) - genie(소수부만).  즉 소수부를 완벽히 알아도")
print("  정수부를 모르면 잃는 양이다.  이것이 심볼 단위로는 회수 불가능한 부분이다.")
