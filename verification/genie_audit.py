# -*- coding: utf-8 -*-
"""
§5.6 (random walk CFO) 의 genie 상한이 **타당한 상한인가**를 검사한다.

쟁점.  drift_baseline.py 는 walk 조건에서 genie - 2D = 22.0%p / 34.8%p 를 보고했고
README 는 그것을 "미회수 여지"로 적었다.  그런데 genie 는 심볼 단위 CFO 궤적을
전부 아는 수신기다.  -25 dB 에서 심볼 하나는 자기 CFO 를 추정할 SNR 자체가 없고
(심볼단위 탐색은 headroom 의 5.0%p 만 회수), 그렇다면 그 22/34%p 는
"여지" 가 아니라 "상한이 느슨한 것" 일 수 있다.  둘은 논문에서 정반대의 주장이다.

세 갈래로 가른다.

  (A) 블록-genie 스윕.  genie 에게 참 궤적을 B 심볼 블록평균으로만 준다.
      B=1 이 현재 genie, B=128 이 패킷-상수 genie.  B 를 키우며 성능이 어디서
      무너지는지가 "궤적을 얼마나 정밀히 알아야 genie 가 되는가" 의 답이다.
      B=1 근처에서만 성능이 나온다면 genie 는 물리적으로 도달 불가능한 상한이다.

  (B) 모델 인지 추적기 (HMM).  아무 side info 없이, 모델만 아는 수신기.
        상태 : CFO 격자 (+-1.5 bin, 0.03 간격)
        우도 : 미지 심볼을 주변화한 것.  log p(y_i | eps) = logsumexp_m log I0(2A|c_m(eps)|/q)
        전이 : 참 증분분산을 아는 가우시안 (walk 면 amp/sqrt(ns))
      전방-후방 스무딩 -> 사후분포 gamma_i(eps) -> eps 를 주변화해 심볼 판정
        m^ = argmax_m  sum_e gamma_i(e) p(y_i | m, e)
      이 추적기가 genie 에 붙으면 walk 의 여지는 없는 것이고, 못 붙으면 여지가 진짜다.

      우도는 두 가지로 계산한다.
        exact : log I0(z)            <- 정확한 비간섭 우도.  이쪽이 주 결과.
        quad  : z^2/4 = A^2|c|^2/q^2 <- 요청받은 |corr|^2/sigma^2 형태.
      여기 z = 2A|c|/q 의 전형값이 ~14 라 소인수 근사가 성립하지 않는다.  약한 근사를
      쓰면 추적기가 약해져서 없는 격차가 생기므로 둘 다 낸다.

  (C) 대조군.  (B) 가 맞게 구현됐는지 독립으로 확인한다.
        C1  amp=0 (정적 CFO) 에서 추적기 == 상수-CFO 패킷 추정기
        C2  선형 drift 에서 추적기 >= drift-aware 2D 추정기
      하나라도 어긋나면 구현 오류다.  그 경우 (A)(B) 의 수치를 쓰면 안 된다.

상관 커널은 drift_baseline.py 와 동일하고 (scores == 4*||x||*|<y,xhat_m>| 을 assert),
CI 는 stats.py 의 packet-cluster / paired bootstrap 이다.

딸린 스크립트 두 개:
  alias_split.py  [A2] genie 우위를 정수부(alias) 몫과 소수부 몫으로 분해한다.
                  본 파일에서 계산은 하지만 표는 거기서 낸다.
  grid_conv.py    C2 의 허용폭 1.5%p 가 상태격자 이산화 몫임을 격자수렴으로 보인다.

첫 실행에서 발견한 것 (기록): 처음에는 HMM 의 사전분포를 격자 전체 균등으로 줬고
대조군이 둘 다 무너졌다 (추적기가 blind 로 퇴화).  원인은 우도가 eps 에 대해
**주기 1 bin 으로 정확히 주기적**이라는 것이다 -- CFO 1 bin 이동이 데이터 심볼
1 칸 순환이동과 같고, 미지 심볼을 균등 주변화하면 그 겹침이 정확해진다.  +-1.5
격자에는 alias 가 3 개 있으므로 균등 사전분포는 셋 중 하나를 잡음으로 고르게 만든다.
참 모델의 사전분포 (eps_0) 를 넣어야 한다.  1D 추정기가 이 문제를 겪지 않았던 것은
격자가 +-0.5, 즉 정확히 한 주기뿐이기 때문이다.
"""
import sys, os, time, json
import numpy as np
import numpy.matlib
from scipy.signal import chirp
from scipy.special import i0e, logsumexp
import scipy.fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import cluster_boot, paired_boot

# ---------------------------------------------------------------- 신호 모델 (drift_baseline.py 와 동일)
sf, bw, OSF = 9, 250_000, 4
fs = bw * OSF
N = 2 ** sf
BINHZ = bw / N


def gen(cw):
    t = np.arange(0, 2 ** sf / bw, 1 / fs); nsm = fs * 2 ** sf / bw
    ci = chirp(t, -bw / 2, 2 ** sf / bw, bw / 2, 'linear', 0)
    cq = chirp(t, -bw / 2, 2 ** sf / bw, bw / 2, 'linear', -90)
    b = numpy.matlib.repmat(ci + 1j * cq, 1, 2); off = round((2 ** sf - cw) / 2 ** sf * nsm)
    return b[:, int(nsm - off):int(nsm - off + int(nsm))][0]


X = np.array([gen(m) for m in range(N)], dtype=np.complex64)
L = X.shape[1]
osf = L // N
t = (np.arange(L) / fs).astype(np.float64)
F0c = np.conj(np.fft.fft(X[0].astype(np.complex128))).astype(np.complex64)
LAG = (-np.arange(N)) % N
SIGPW = float(np.mean(np.abs(X[0]) ** 2))
AMP = float(np.linalg.norm(X[0]))            # ||x||
P = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.complex64)

NS = 128
NPK = int(os.environ.get('NPK', 120))
SNR = -25
QN = SIGPW / (10 ** (SNR / 10))              # 표본당 잡음 전력
GSNR = AMP ** 2 / QN                         # 상관 후 SNR = A^2/q
IRAMP = np.linspace(-0.5, 0.5, NS)
G1 = np.arange(-0.5, 0.501, 0.125)           # 1D coarse (drift_baseline 과 동일)
G1F = np.round(np.arange(-16, 17) * 0.03125, 6)   # 1D fine, +-0.5 / 0.03125
GS = np.linspace(-2.0, 2.0, 9)               # 2D slope
BLIST = [1, 2, 4, 8, 16, 32, 64, 128]
EGRID = {"spec (+-1.5, 0.03, 101)": np.round(np.arange(-50, 51) * 0.03, 6),
         "wide (+-3.0, 0.03, 201)": np.round(np.arange(-100, 101) * 0.03, 6)}
COVER_TOL = 0.005                            # 격자 밖 참 CFO 비율이 이보다 크면 wide 도 돌린다

rng = np.random.default_rng(90210)


def ramp(eps):
    return np.exp(-2j * np.pi * (np.asarray(eps, float)[:, None] * BINHZ) * t[None, :]
                  ).astype(np.complex64)


def scores(Y):
    """|<y_i, x_m>| * 4.  drift_baseline.scores 와 동일 (아래에서 assert)."""
    Z = sfft.fft(Y, axis=1, workers=-1) * F0c[None, :]
    return np.abs(sfft.ifft(Z.reshape(len(Y), osf, N).sum(1), axis=1, workers=-1))[:, LAG]


def scores_ref(Y):
    Z = np.fft.fft(Y, axis=1) * F0c[None, :]
    return np.abs(np.fft.ifft(Z.reshape(len(Y), osf, N).sum(1), axis=1))[:, LAG]


def rx(syms, eps):
    Y = X[syms] * np.conj(ramp(eps))
    return Y + (rng.normal(0, np.sqrt(QN / 2), Y.shape)
                + 1j * rng.normal(0, np.sqrt(QN / 2), Y.shape)).astype(np.complex64)


V_OFF = np.stack([ramp(np.array([f]))[0] for f in G1])
V_OFFF = np.stack([ramp(np.array([f]))[0] for f in G1F])
V_SLP = np.stack([ramp(g * IRAMP) for g in GS])
V_E = {k: np.stack([ramp(np.array([e]))[0] for e in g]) for k, g in EGRID.items()}

# ---------------------------------------------------------------- sanity: 커널 등가성 + 스케일
_s = rng.integers(0, N, 64)
_Y = rx(_s, np.zeros(64))
assert (scores(_Y).argmax(1) == scores_ref(_Y).argmax(1)).all(), "scipy-fft != numpy-fft 판정"
_r = scores(_Y) / (4 * AMP * np.abs(_Y @ P.conj().T))
assert abs(_r.mean() - 1) < 1e-4 and _r.std() < 1e-4, f"scale 틀림 {_r.mean()} {_r.std()}"
print(f"sanity: scores == 4*||x||*|<y,xhat_m>|  (ratio {_r.mean():.6f} +- {_r.std():.1e}),"
      f"  scipy-fft 판정 == numpy-fft 판정")
print(f"A^2/q = 상관후 SNR = {GSNR:.3f} ({10*np.log10(GSNR):.2f} dB),  "
      f"z = 2A|c|/q 전형값 ~ {2*np.sqrt(GSNR*(1+GSNR)):.1f}  -> log I0 의 소인수근사 z^2/4 는 무효\n")


# ---------------------------------------------------------------- 고전 arm 들
def dec_blind(Yp):
    return scores(Yp).argmax(1)


def dec_const(Yp, V):
    bi, bs = 0, -np.inf
    for i in range(len(V)):
        v = scores(Yp * V[i][None, :]).max(1).sum()
        if v > bs:
            bs, bi = v, i
    return scores(Yp * V[bi][None, :]).argmax(1)


def dec_drift(Yp):
    best, bs = (0, 0), -np.inf
    for j in range(len(GS)):
        Yg = Yp * V_SLP[j]
        for i in range(len(G1)):
            v = scores(Yg * V_OFF[i][None, :]).max(1).sum()
            if v > bs:
                bs, best = v, (i, j)
    return scores(Yp * V_SLP[best[1]] * V_OFF[best[0]][None, :]).argmax(1)


def dec_genie(Yp, eps):
    return scores(Yp * ramp(eps)).argmax(1)


def blk(eps, B):
    return np.repeat(eps.reshape(-1, B).mean(1), B)


# ---------------------------------------------------------------- (B) HMM 추적기
def lam_tensor(Yp, VE):
    """lam[e,i,m] = |<y_i, xhat_m(eps_e)>|^2 / q   (잡음전력으로 정규화한 상관 전력)"""
    ne = len(VE)
    out = np.empty((ne, len(Yp), N), np.float32)
    for e in range(ne):
        sc = scores(Yp * VE[e][None, :])
        out[e] = ((sc / (4 * AMP)) ** 2 / QN).astype(np.float32)
    return out


def metric(lam, kind):
    """심볼별 log p(y | m, eps) (eps-무관 상수 제외)."""
    if kind == "exact":
        z = (2 * np.sqrt(GSNR) * np.sqrt(lam)).astype(np.float64)    # z = 2A|c|/q
        return (np.log(i0e(z)) + z).astype(np.float32)
    return (GSNR * lam).astype(np.float32)                           # z^2/4 = A^2|c|^2/q^2


def prior_eps0(kind, amp, E):
    """eps_0 의 참 주변분포를 격자에 올린다.

    우도는 eps 에 대해 **주기 1 bin 으로 정확히 주기적**이다 (CFO 1 bin 이동 ==
    데이터 심볼 1 칸 순환이동, 미지 심볼을 균등 주변화하면 그 겹침이 정확해진다).
    따라서 +-1.5 격자에는 alias 가 3 개 있고, 균등 사전분포를 주면 추적기가
    셋 중 하나를 잡음으로 고른다.  참 모델의 사전분포 (base ~ U(-0.5,0.5)) 가
    그 겹침을 깨는 유일한 정보다.  이것을 빼먹으면 추적기는 blind 로 퇴화한다.
    """
    pr = np.random.default_rng(20240915)
    M = 60000
    b0 = pr.uniform(-0.5, 0.5, M)
    if kind == "linear":
        e0 = b0 + amp * IRAMP[0]
    elif amp > 0:
        w = np.cumsum(pr.normal(0, amp / np.sqrt(NS), (M, NS)), axis=1)
        e0 = b0 + w[:, 0] - w.mean(1)
    else:
        e0 = b0
    step = E[1] - E[0]
    edges = np.concatenate([E - step / 2, [E[-1] + step / 2]])
    h, _ = np.histogram(e0, bins=edges)
    p = h.astype(float)
    if p.sum() == 0:
        p[:] = 1.0
    p /= p.sum()
    return p


def fwd_bwd(LL, E, sig, pri, mu=0.0):
    """LL: (ns, ne) 심볼별 log p(y_i|eps).  반환 gamma (ns, ne).

    전이는 평균 mu, 표준편차 sig 의 가우시안이다.  walk 는 (mu=0, sig=amp/sqrt(ns)),
    선형 drift 는 증분이 결정론적이므로 (mu=amp/ns, sig=격자간격/2) 가 참 모델이다.
    """
    ns, ne = LL.shape
    if sig <= 0 and mu == 0.0:
        T = np.eye(ne)
    else:
        d = E[None, :] - E[:, None] - mu
        T = np.exp(-0.5 * (d / max(sig, 1e-9)) ** 2)
        T /= np.maximum(T.sum(1, keepdims=True), 1e-300)
    lik = np.exp(LL - LL.max(1, keepdims=True))
    a = np.empty((ns, ne)); b = np.empty((ns, ne))
    p = pri.copy()
    for i in range(ns):
        v = p * lik[i]; s = v.sum(); a[i] = v / (s if s > 0 else 1.0); p = a[i] @ T
    b[ns - 1] = 1.0
    for i in range(ns - 2, -1, -1):
        v = T @ (lik[i + 1] * b[i + 1]); s = v.sum(); b[i] = v / (s if s > 0 else 1.0)
    g = a * b
    return g / g.sum(1, keepdims=True)


def track(lam, E, sig, kind, pri, mu=0.0):
    """반환 (심볼판정, 사후평균 eps)."""
    G = metric(lam, kind)
    LL = logsumexp(G, axis=2).T                       # (ns, ne)
    gam = fwd_bwd(LL, E, sig, pri, mu)
    ref = G.max(axis=(0, 2))
    W = np.exp(G - ref[None, :, None])
    S = np.einsum('eim,ie->im', W, gam.astype(np.float32), optimize=True)
    return S.argmax(1), gam @ E


# ---------------------------------------------------------------- 조건 실행
def run(kind, amp, want_wide):
    grids = [list(EGRID)[0]] + ([list(EGRID)[1]] if want_wide else [])
    arms = {}

    def slot(k):
        return arms.setdefault(k, np.zeros(NPK * NS, bool))

    out_cov = 0
    rmse = {g: [] for g in grids}
    step = float(EGRID[grids[0]][1] - EGRID[grids[0]][0])
    if kind == "linear":
        # 참 모델: 증분이 결정론적 (amp/ns).  sig 는 격자 이산화 몫만.
        TRANS = [("", amp / NS, step / 2),
                 (",rw", 0.0, amp / np.sqrt(NS))]   # 참고: 오설정 random-walk 모델
    else:
        TRANS = [("", 0.0, (amp / np.sqrt(NS)) if amp > 0 else 0.0)]
    PRI = {gn: prior_eps0(kind, amp, EGRID[gn]) for gn in grids}
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
        out_cov += int((np.abs(eps) > 1.5).sum())
        slot("blind MF")[sl] = dec_blind(Yp) == s
        slot("1D coarse")[sl] = dec_const(Yp, V_OFF) == s
        slot("1D fine")[sl] = dec_const(Yp, V_OFFF) == s
        slot("2D drift-aware")[sl] = dec_drift(Yp) == s
        for B in BLIST:
            slot(f"genie B={B}")[sl] = dec_genie(Yp, blk(eps, B)) == s
        nint = np.round(eps)
        slot("genie 정수부만")[sl] = dec_const(Yp * ramp(nint), V_OFFF) == s
        slot("genie 소수부만")[sl] = dec_genie(Yp, eps - nint) == s
        for gname in grids:
            E = EGRID[gname]
            lam = lam_tensor(Yp, V_E[gname])
            for kd in ("exact", "quad"):
                for tag, mu, sg in TRANS:
                    mh, eh = track(lam, E, sg, kd, PRI[gname], mu)
                    slot(f"tracker {kd} [{gname.split()[0]}{tag}]")[sl] = mh == s
                    if kd == "exact" and tag == "":
                        rmse[gname].append(float(np.sqrt(np.mean((eh - eps) ** 2))))
            del lam
    return arms, out_cov / (NPK * NS), {g: float(np.mean(v)) for g, v in rmse.items()}


def ci(a):
    m, lo, hi, _ = cluster_boot(a, NS, seed=7)
    return m, lo, hi


def pb(a, b):
    return paired_boot(a, b, NS, seed=7)


CONDS = [("walk", 0.0), ("walk", 0.5), ("walk", 1.0), ("walk", 2.0),
         ("linear", 1.0), ("linear", 2.0)]
print(f"SNR = {SNR} dB,  ns = {NS},  {NPK} packets x {NS} = {NPK*NS} symbols/cell")
print(f"추적기 격자 = {list(EGRID)[0]}, 전이 sigma = amp/sqrt(ns)\n")

RES = {}
for kind, amp in CONDS:
    t0 = time.perf_counter()
    # 격자 밖 비율을 먼저 싸게 추정해 wide 필요 여부를 정한다
    pr = np.random.default_rng(1234)
    oc = 0
    for _ in range(400):
        b0 = pr.uniform(-0.5, 0.5)
        if kind == "linear":
            e0 = b0 + amp * IRAMP
        else:
            w = np.cumsum(pr.normal(0, amp / np.sqrt(NS), NS)) if amp > 0 else np.zeros(NS)
            e0 = b0 + w - w.mean()
        oc += int((np.abs(e0) > 1.5).sum())
    want_wide = oc / (400 * NS) > COVER_TOL
    arms, cov, rmse = run(kind, amp, want_wide)
    RES[f"{kind}:{amp}"] = dict(arms=arms, cover_out=cov, rmse=rmse, wide=want_wide)
    print(f"[{kind} amp={amp}]  done {time.perf_counter()-t0:.0f}s   "
          f"참 CFO 가 +-1.5 격자 밖인 심볼 = {cov*100:.2f}%   "
          f"wide 격자 추가 = {want_wide}   "
          + "  ".join(f"RMSE[{g.split()[0]}]={v:.3f}bin" for g, v in rmse.items()))
    sys.stdout.flush()

A = {k: d["arms"] for k, d in RES.items()}
BAR = "=" * 120

# ---------------------------------------------------------------- (C) 대조군 먼저
print("\n" + BAR + "\n[C] 대조군 — 여기서 어긋나면 (A)(B) 의 수치를 쓰면 안 된다\n" + BAR)
GATE = {}
a0 = A["walk:0.0"]
tn0 = "tracker exact [spec]"
d, lo, hi, p = pb(a0[tn0], a0["1D fine"])
GATE["C1"] = bool((lo <= 0 <= hi) or abs(d) < 1.0)
print(f"  C1  정적 CFO(amp=0): 추적기 - 1D fine  = {d:+6.2f}%p [{lo:6.2f},{hi:6.2f}] p={p:.4f}"
      f"   -> {'PASS' if GATE['C1'] else 'FAIL'}   (기준: 0 을 포함하거나 |차이|<1.0%p)")
d, lo, hi, p = pb(a0[tn0], a0["1D coarse"])
print(f"      참고: 추적기 - 1D coarse = {d:+6.2f}%p [{lo:6.2f},{hi:6.2f}] p={p:.4f}"
      f"  (격자 0.03 vs 0.125 의 양자화 몫만큼은 벌어질 수 있다)")
ok2 = True
for amp in (1.0, 2.0):
    ax = A[f"linear:{amp}"]
    d, lo, hi, p = pb(ax["tracker exact [spec]"], ax["2D drift-aware"])
    # 허용폭 1.5%p: 상태격자 이산화 몫.  _diag2.py 에서 격자간격 0.06->0.0075 로
    # 좁히면 RMSE 가 0.067->0.029 bin 으로 단조 수렴하고 추적기-2D 가 0 근방에
    # 머무는 것을 확인했다 (즉 남는 잔차는 구현이 아니라 격자 몫이다).
    good = bool(hi >= 0 or abs(d) < 1.5)
    ok2 = ok2 and good
    print(f"  C2  선형 drift {amp} bin: 추적기 - 2D = {d:+6.2f}%p [{lo:6.2f},{hi:6.2f}] p={p:.4f}"
          f"   -> {'PASS' if good else 'FAIL'}   (기준: CI 상한 >= 0 또는 |차이| < 1.5%p = 격자 이산화 허용폭)")
GATE["C2"] = ok2
print(f"\n  GATE = {'PASS' if all(GATE.values()) else 'FAIL'}  {GATE}")

# ---------------------------------------------------------------- (A) 블록-genie
print("\n" + BAR + "\n[A] 블록-genie 스윕 — genie 에게 참 궤적을 B 심볼 블록평균으로만 준다"
      "\n    B=1 이 현재 genie, B=128 이 패킷-상수 genie\n" + BAR)
print(f"  {'조건':>14} |" + "".join(f"{'B='+str(B):>10}" for B in BLIST)
      + f"{'blind':>10}{'2D':>10}")
for kind, amp in CONDS:
    a = A[f"{kind}:{amp}"]
    row = f"  {kind+' '+format(amp,'.2f'):>14} |"
    for B in BLIST:
        row += f"{ci(a[f'genie B={B}'])[0]:>9.1f}%"
    row += f"{ci(a['blind MF'])[0]:>9.1f}%{ci(a['2D drift-aware'])[0]:>9.1f}%"
    print(row)
print("\n  같은 표에 CI 와 B=1 대비 손실:")
for kind, amp in CONDS:
    a = A[f"{kind}:{amp}"]
    print(f"  [{kind} amp={amp}]")
    for B in BLIST:
        m, lo, hi = ci(a[f"genie B={B}"])
        if B == 1:
            print(f"      B={B:>3}  {m:5.1f}% [{lo:4.1f},{hi:4.1f}]   (= 현재 genie)")
        else:
            d, dlo, dhi, p = pb(a["genie B=1"], a[f"genie B={B}"])
            print(f"      B={B:>3}  {m:5.1f}% [{lo:4.1f},{hi:4.1f}]   "
                  f"B=1 - B={B} = {d:+6.2f}%p [{dlo:6.2f},{dhi:6.2f}] p={p:.4f}")

# ---------------------------------------------------------------- (B) 추적기
print("\n" + BAR + "\n[B] 모델 인지 HMM 추적기 — side info 없음, 모델만 앎\n" + BAR)
print(f"  {'조건':>14} |{'blind':>22}{'1D':>22}{'2D':>22}{'tracker(exact)':>22}"
      f"{'genie(B=1)':>22}")
for kind, amp in CONDS:
    a = A[f"{kind}:{amp}"]
    row = f"  {kind+' '+format(amp,'.2f'):>14} |"
    for nm in ["blind MF", "1D coarse", "2D drift-aware", "tracker exact [spec]", "genie B=1"]:
        m, lo, hi = ci(a[nm])
        row += f"{m:>11.1f}% [{lo:4.1f},{hi:4.1f}]"
    print(row)
print()
for kind, amp in CONDS:
    a = A[f"{kind}:{amp}"]
    print(f"  [{kind} amp={amp}]")
    for nm in sorted(n for n in a if n.startswith("tracker")):
        m, lo, hi = ci(a[nm])
        dg, glo, ghi, gp = pb(a["genie B=1"], a[nm])
        d2, l2, h2, p2 = pb(a[nm], a["2D drift-aware"])
        dp, lp, hp, pp = pb(a[nm], a["genie B=128"])
        print(f"      {nm:<26} {m:5.1f}% [{lo:4.1f},{hi:4.1f}]  "
              f"genie-tr = {dg:+6.2f}%p [{glo:6.2f},{ghi:6.2f}] p={gp:.4f}  "
              f"tr-2D = {d2:+6.2f}%p [{l2:6.2f},{h2:6.2f}]  "
              f"tr-genieB128 = {dp:+6.2f}%p [{lp:6.2f},{hp:6.2f}]")

# ---------------------------------------------------------------- 요약
print("\n" + BAR + "\n[요약] README §5.6 의 '미회수 여지' 가 실제로 얼마나 남는가\n" + BAR)
print(f"  {'조건':>14} |{'genie - 2D (기존 주장)':>30}{'genie - tracker (실제 여지)':>36}"
      f"{'tracker - 2D (추적기 회수분)':>36}")
for kind, amp in CONDS:
    a = A[f"{kind}:{amp}"]
    tn = "tracker exact [spec]"
    d1, l1, h1, _ = pb(a["genie B=1"], a["2D drift-aware"])
    d2, l2, h2, p2 = pb(a["genie B=1"], a[tn])
    d3, l3, h3, p3 = pb(a[tn], a["2D drift-aware"])
    print(f"  {kind+' '+format(amp,'.2f'):>14} |{d1:>16.2f}%p [{l1:5.2f},{h1:5.2f}]"
          f"{d2:>18.2f}%p [{l2:5.2f},{h2:5.2f}] p={p2:.3f}"
          f"{d3:>16.2f}%p [{l3:5.2f},{h3:5.2f}] p={p3:.3f}")
print(f"\n  GATE(C) = {'PASS' if all(GATE.values()) else 'FAIL'} -> "
      f"{'위 수치 사용 가능' if all(GATE.values()) else '구현 오류. 수치를 쓰지 말 것.'}")

_out = {k: {"cover_out": v["cover_out"], "rmse": v["rmse"], "wide": v["wide"],
            "acc": {n: float(np.mean(a)) * 100 for n, a in v["arms"].items()}}
        for k, v in RES.items()}
_out["gate"] = {k: bool(v) for k, v in GATE.items()}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "genie_audit_out.json"), "w", encoding="utf-8") as f:
    json.dump(_out, f, indent=1, ensure_ascii=False)
print("\nraw -> verification/genie_audit_out.json")
