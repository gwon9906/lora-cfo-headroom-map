# -*- coding: utf-8 -*-
"""
§5.7 이 가정한 "초기 정수 CFO 는 해소됐다"를 실제로 재서 검증한다.

§5.7 (genie_audit.py) 은 패킷 시작 CFO 를 base ~ U(-0.5, 0.5) 로 둔다.  이는
**프리앰블이 정수 CFO 를 완벽히 해소했다**는 모델이다.  리뷰어가 물을 지점이 정확히
여기다: "-25 dB 에서 프리앰블이 정말 그걸 할 수 있나?"

sigma_pre 를 가정하지 않는다.  실제 프리앰블 추정기를 만들어서 잰다.

  프리앰블  : LoRa 표준대로 up-chirp 8 개.  심볼이 0 으로 **기지**이므로
              §5.7 의 1 bin alias 가 여기엔 없다 -- 심볼을 주변화하지 않기 때문이다.
              그래서 프리앰블은 원리적으로 정수부를 잡을 수 있다.  문제는 SNR 뿐이다.
  추정기    : eps in [-2, 2] bin 격자에서 sum_k |<y_k(eps), xhat_0>|^2 최대화.
              심볼 간 위상 연속성은 이 저장소 모델에 없으므로(각 심볼의 CFO 위상이
              0 에서 다시 시작한다) 비간섭 결합만 쓴다.  실제 수신기가 하는 것과 같다.
  측정      : 잔차 r = eps_payload[0] - eps_hat.
              std(r) 와 **P(|r| > 0.5) = 정수부 오류율**.  후자가 핵심이다.

그 다음 페이로드를 eps_hat 으로 미리 보상하고 §5.7 의 arm 을 다시 돌린다.
genie 는 참 궤적을 알므로 영향받지 않는다.  따라서

  - 잔차가 작으면  -> §5.7 의 결론이 완전 동기 가정에 의존하지 않는다는 뜻
  - 잔차가 크면    -> 프리앰블조차 정수부를 못 잡는다는 뜻이고, 그건 §5.7 의
                      주장을 **더 강하게** 만든다 (여지가 오히려 커진다)

어느 쪽이든 결과가 나온다.  그래서 이 실험은 안전하다.

전제: 심볼 타이밍은 정확히 안다고 본다 (저장소 전체가 그렇다).  실제 LoRa 는
down-chirp 2.25 개로 CFO 와 STO 를 분리하는데, 여기서는 STO 가 0 이므로
그 단계를 생략하고 CFO 만 추정한다.  논문에서 명시해야 할 가정이다.
"""
import os, io, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
src = io.open(os.path.join(HERE, "genie_audit.py"), encoding="utf-8").read()
head = src.split("CONDS = [")[0]
g = {"__name__": "preamble_residual", "__file__": os.path.join(HERE, "genie_audit.py")}
exec(compile(head, "genie_audit_head", "exec"), g)

np_ = np
NS, N, L = g["NS"], g["N"], g["L"]
X, P, AMP, QN, GSNR = g["X"], g["P"], g["AMP"], g["QN"], g["GSNR"]
IRAMP, EGRID, V_E = g["IRAMP"], g["EGRID"], g["V_E"]
V_OFF, V_SLP, V_OFFF = g["V_OFF"], g["V_SLP"], g["V_OFFF"]
ramp, scores = g["ramp"], g["scores"]
dec_blind, dec_const, dec_drift, dec_genie = (g["dec_blind"], g["dec_const"],
                                              g["dec_drift"], g["dec_genie"])
lam_tensor, track = g["lam_tensor"], g["track"]
cluster_boot, paired_boot = g["cluster_boot"], g["paired_boot"]

NPK = int(os.environ.get("NPK", 120))
NPRE = 8                                        # LoRa 표준 프리앰블 up-chirp 개수
GNAME = "wide (+-3.0, 0.03, 201)"
EG = EGRID[GNAME]

# 프리앰블 CFO 격자: +-2 bin, 0.025 간격.  ramp(e)[l] = exp(-2i*pi*e*l/L) 이므로
# 상관은 u_k 의 비정수 주파수 DFT 다.  161x2048 행렬 하나면 끝난다.
# 탐색 범위 주의.  프리앰블은 **패킷 시작** CFO 를 준다.  §5.7 은 페이로드 평균을
# base 로 뒀는데 그건 "평균을 아는" 유리한 규약이었다.  실제로는 시작점만 알고
# 거기서 드리프트한다.  그래서 보상 후 잔차가 0 에서 한쪽으로 램프한다 (선형 2 bin
# 이면 0 -> +1.9).  모든 arm 의 격자를 그 범위까지 넓혀야 공정한 비교가 된다.
# 좁은 격자로 재면 물리가 아니라 격자 어긋남을 재게 된다.
PG1 = np.round(np.arange(-12, 13) * 0.125, 6)          # 2D offset: +-1.5
PGS = np.linspace(-2.5, 2.5, 11)                       # 2D slope: +-2.5
PG = np.round(np.arange(-80, 81) * 0.025, 6)
_l = np.arange(L)
RAMPS = np.exp(-2j * np.pi * PG[:, None] * _l[None, :] / L).astype(np.complex64)
X0H = np.conj(P[0]).astype(np.complex64)        # 정규화된 프리앰블 심볼 (심볼 0)
V_OFF2 = np.stack([ramp(np.array([f]))[0] for f in PG1])
V_SLP2 = np.stack([ramp(v * IRAMP) for v in PGS])


# 어떤 arm 이든 탐색폭이 1 bin 을 넘으면 alias 에 걸린다 (offset 을 1 올리면 모든
# 심볼이 1 칸 밀린다).  추적기는 프리앰블 사전분포로 그걸 깬다.  고전 arm 도 같은
# 정보를 줘야 공정하다 -> **궤적의 시작점이 0 근처여야 한다**는 제약을 건다.
# 2D 모델 offset + slope*IRAMP 의 시작값은 offset - slope/2 이므로:
PRE_OK = [(i, j) for j in range(len(PGS)) for i in range(len(PG1))
          if abs(PG1[i] - PGS[j] / 2) <= 0.5]


def dec_drift2(Yp):
    """offset x slope 2D 탐색, 프리앰블이 시작점을 +-0.5 로 고정한 것을 반영."""
    best, bs = PRE_OK[0], -np.inf
    cache = {}
    for i, j in PRE_OK:
        if j not in cache:
            cache = {j: Yp * V_SLP2[j]}
        v = scores(cache[j] * V_OFF2[i][None, :]).max(1).sum()
        if v > bs:
            bs, best = v, (i, j)
    return scores(Yp * V_SLP2[best[1]] * V_OFF2[best[0]][None, :]).argmax(1)


def est_preamble(Ypre):
    """비간섭 결합 프리앰블 CFO 추정.  반환 eps_hat (bin)."""
    u = Ypre * X0H[None, :]                     # (NPRE, L)
    C = u @ RAMPS.T                             # (NPRE, |PG|)
    return float(PG[(np.abs(C) ** 2).sum(0).argmax()])


def traj(kind, amp, base, rng):
    """프리앰블 + 페이로드를 관통하는 CFO 궤적.  페이로드 평균이 base 가 되도록
    맞춰 §5.7 의 규약과 일치시킨다."""
    tot = NPRE + NS
    if kind == "linear":
        e = base + amp * np.linspace(-0.5, 0.5, tot)
        e = e - e[NPRE:].mean() + base
    elif amp > 0:
        w = np.cumsum(rng.normal(0, amp / np.sqrt(NS), tot))
        e = base + w - w[NPRE:].mean()
    else:
        e = np.full(tot, base)
    return e[:NPRE], e[NPRE:]


def rx_block(syms, eps, rng):
    Y = X[syms] * np.conj(ramp(eps))
    return Y + (rng.normal(0, np.sqrt(QN / 2), Y.shape)
                + 1j * rng.normal(0, np.sqrt(QN / 2), Y.shape)).astype(np.complex64)


CONDS = [("walk", 0.0), ("walk", 1.0), ("walk", 2.0), ("linear", 2.0)]
BAR = "=" * 118
print(f"SNR = -25 dB, ns = {NS}, 프리앰블 {NPRE} up-chirps, {NPK} packets/cell")
print(f"프리앰블 추정 격자 = +-2.0 bin / 0.025 ({len(PG)} 가설), 비간섭 결합")
print(f"페이로드 arm: 1D offset +-0.5 (시작점 제약), "
      f"2D = {len(PRE_OK)} 가설 (프리앰블 제약 통과분), 추적기 {GNAME}")
print(f"심볼당 상관후 SNR = {10*np.log10(GSNR):.2f} dB, "
      f"프리앰블 {NPRE} 개 풀링 = {10*np.log10(GSNR*NPRE):.2f} dB\n")

# ---------------------------------------------------------------- 1차: 잔차 분포 측정
print(BAR + "\n[1] 프리앰블 잔차 측정 — sigma 를 가정하지 않고 직접 잰다\n" + BAR)
print(f"  {'조건':>13} |{'std(r)':>12}{'|r| 중앙값':>13}"
      f"{'P(|r|>0.5) = 정수부 오류':>26}{'P(|r|>0.15)':>14}")
RES = {}
for kind, amp in CONDS:
    rng = np.random.default_rng(4242)
    r = np.empty(NPK * 3)
    for k in range(len(r)):
        base = float(np.round(rng.uniform(-0.5, 0.5), 3))
        epre, epay = traj(kind, amp, base, rng)
        Ypre = rx_block(np.zeros(NPRE, int), epre, rng)
        r[k] = epay[0] - est_preamble(Ypre)
    RES[f"{kind}:{amp}"] = r
    print(f"  {kind+' '+format(amp,'.2f'):>13} |{r.std():>12.4f}{np.median(np.abs(r)):>13.4f}"
          f"{(np.abs(r) > 0.5).mean()*100:>25.2f}%{(np.abs(r) > 0.15).mean()*100:>13.2f}%")

# 추적기 사전분포를 측정된 잔차 분포에서 만든다 (모델 인지의 연장)
def prior_from(r, E):
    step = E[1] - E[0]
    edges = np.concatenate([E - step / 2, [E[-1] + step / 2]])
    h, _ = np.histogram(r, bins=edges)
    p = h.astype(float)
    p = p + p.max() * 1e-3 if p.sum() > 0 else np.ones(len(E))
    return p / p.sum()


# ---------------------------------------------------------------- 2차: 페이로드 재측정
print("\n" + BAR + "\n[2] 프리앰블로 미리 보상한 뒤 §5.7 의 arm 재측정\n" + BAR)
ARMS = ["blind MF", "1D", "2D", "tracker", "genie"]
OUT = {}
for kind, amp in CONDS:
    t0 = time.perf_counter()
    rng = np.random.default_rng(90210)
    pri = prior_from(RES[f"{kind}:{amp}"], EG)
    sig = (amp / np.sqrt(NS)) if amp > 0 else 0.0
    mu = (amp / NS) if kind == "linear" else 0.0
    if kind == "linear":
        sig = float(EG[1] - EG[0]) / 2
    ok = {a: np.zeros(NPK * NS, bool) for a in ARMS}
    nclip = 0
    nfar = 0
    for k in range(NPK):
        s = rng.integers(0, N, NS)
        base = float(np.round(rng.uniform(-0.5, 0.5), 3))
        epre, epay = traj(kind, amp, base, rng)
        Ypre = rx_block(np.zeros(NPRE, int), epre, rng)
        ehat = est_preamble(Ypre)
        Ypay = rx_block(s, epay, rng)
        Yc = Ypay * ramp(np.full(NS, ehat))     # 프리앰블 추정으로 사전 보상
        res = epay - ehat                        # 남은 잔차 궤적
        nclip += int((np.abs(res) > abs(EG[0])).sum())
        nfar += int((np.abs(res) > 1.5).sum())
        sl = slice(k * NS, (k + 1) * NS)
        ok["blind MF"][sl] = dec_blind(Yc) == s
        ok["1D"][sl] = dec_const(Yc, V_OFF) == s
        ok["2D"][sl] = dec_drift2(Yc) == s
        ok["genie"][sl] = dec_genie(Yc, res) == s
        lam = lam_tensor(Yc, V_E[GNAME])
        mh, _ = track(lam, EG, sig, "exact", pri, mu)
        ok["tracker"][sl] = mh == s
        del lam
    OUT[f"{kind}:{amp}"] = ok
    print(f"  [{kind} amp={amp}] {time.perf_counter()-t0:.0f}s  "
          f"잔차가 추적기 격자({EG[0]:+.1f}~{EG[-1]:+.1f}) 밖 = {nclip/(NPK*NS)*100:.2f}%"
          f",  |잔차|>1.5 bin = {nfar/(NPK*NS)*100:.2f}%")
    sys.stdout.flush()

# ---------------------------------------------------------------- 비교
REF = {"walk:0.0": (52.2, 52.3, 0.08), "walk:1.0": (30.4, 52.5, 22.12),
       "walk:2.0": (20.6, 52.7, 32.10), "linear:2.0": (52.1, 53.0, 0.90)}
print("\n" + BAR + "\n[3] §5.7 (완전 동기 가정) 대비\n" + BAR)
print(f"  {'조건':>13} |{'blind':>10}{'1D':>10}{'2D':>10}{'추적기':>20}{'genie':>10}"
      f"{'genie - 추적기':>24}{'§5.7 값':>12}{'변화':>10}")
for kind, amp in CONDS:
    key = f"{kind}:{amp}"
    a = OUT[key]
    row = f"  {kind+' '+format(amp,'.2f'):>13} |"
    for nm in ["blind MF", "1D", "2D"]:
        row += f"{cluster_boot(a[nm], NS, seed=7)[0]:>9.1f}%"
    m, lo, hi, _ = cluster_boot(a["tracker"], NS, seed=7)
    row += f"{m:>11.1f}% [{lo:4.1f},{hi:4.1f}]"
    row += f"{cluster_boot(a['genie'], NS, seed=7)[0]:>9.1f}%"
    d, dlo, dhi, p = paired_boot(a["genie"], a["tracker"], NS, seed=7)
    ref = REF[key][2]
    row += f"{d:>14.2f}%p [{dlo:5.2f},{dhi:5.2f}]{ref:>11.2f}%p{d-ref:>+9.2f}"
    print(row)

print("\n  '변화' 가 0 근방이면 §5.7 의 결론이 완전 동기 가정에 의존하지 않는다.")
print("  양수로 크면 프리앰블조차 못 잡는다는 뜻이고, 여지는 오히려 커진다.")

with open(os.path.join(HERE, "preamble_residual_out.json"), "w", encoding="utf-8") as f:
    json.dump({k: {"resid_std": float(RES[k].std()),
                   "int_err_rate": float((np.abs(RES[k]) > 0.5).mean()),
                   "acc": {n: float(np.mean(v)) * 100 for n, v in OUT[k].items()}}
               for k in OUT}, f, indent=1, ensure_ascii=False)
print("\nraw -> verification/preamble_residual_out.json")
