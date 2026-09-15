# -*- coding: utf-8 -*-
"""
CFO 실험 전체를 신뢰구간과 함께 재측정한다.

바뀐 점 두 가지:
 1) 상관을 direct MF bank (512x2048 MAC) 대신 FFT-MF (2048-FFT + 512-IFFT) 로 계산한다.
    folding.py [0]/[3] 에서 두 구현의 판정이 정확히 일치함을 확인했고, 여기서도 다시
    assert 한다.  ~60배 빨라지므로 trial 수를 40배로 늘릴 수 있다.
 2) 구간추정을 붙인다.  단 두 종류를 구분한다:
      - i.i.d. 심볼 시행 -> Wilson score interval
      - 패킷 pooling (CFO 를 패킷당 한 번 추정) -> 패킷이 클러스터다.
        심볼 단위 이항 CI 는 유효 표본수를 과대평가하므로 packet cluster bootstrap 을 쓴다.
    특히 [D] 의 "상수-CFO 추정기 < blind MF" 주장은 **대응표본(paired) 부트스트랩**으로
    차이의 CI 를 직접 낸다.  같은 패킷·같은 잡음에서 두 방식을 비교하므로 짝지을 수 있다.
"""
import sys, os
import numpy as np
import numpy.matlib
from scipy.signal import chirp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import wilson, cluster_boot, paired_boot

rng = np.random.default_rng(31337)
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
P = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.complex64)
t = (np.arange(L) / fs).astype(np.float64)
F0c = np.conj(np.fft.fft(X[0].astype(np.complex128))).astype(np.complex64)
LAG = (-np.arange(N)) % N
SIGPW = float(np.mean(np.abs(X[0]) ** 2))


def scores(Y, f=0.0):
    """CFO 가설 f (bin) 로 회전시킨 프로토타입 뱅크와의 상관 크기. (n,512)"""
    if f != 0.0:
        Y = Y * np.exp(-2j * np.pi * (f * BINHZ) * t[None, :]).astype(np.complex64)
    Z = np.fft.fft(Y, axis=1) * F0c[None, :]
    A = np.abs(np.fft.ifft(Z.reshape(len(Y), osf, N).sum(1), axis=1))
    return A[:, LAG]


def rx(syms, snr_db, eps):
    Y = X[syms] * np.exp(2j * np.pi * (eps[:, None] * BINHZ) * t[None, :]).astype(np.complex64)
    q = SIGPW / (10 ** (snr_db / 10))
    return Y + (rng.normal(0, np.sqrt(q / 2), Y.shape)
                + 1j * rng.normal(0, np.sqrt(q / 2), Y.shape)).astype(np.complex64)


# ---- sanity: FFT-MF == direct MF bank -----------------------------------------
_s = rng.integers(0, N, 400)
_Y = rx(_s, -22, np.zeros(400))
_a = scores(_Y).argmax(1)
_b = np.abs(_Y @ P.conj().T).argmax(1)
assert (_a == _b).mean() == 1.0, "FFT-MF != direct MF"
print("sanity: FFT-MF 판정 == direct MF bank 판정  (400/400 일치)\n")

GRID = np.arange(-0.5, 0.501, 0.125)     # 9 hypotheses
BAR = "=" * 96


def head(s):
    print("\n" + BAR + "\n" + s + "\n" + BAR)


# ------------------------------------------------------------------ helpers
def ok_blind(Y, s):
    return scores(Y).argmax(1) == s


def ok_genie(Y, s, eps):
    """심볼마다 참 CFO 를 알고 보상 (상한).  eps 가 심볼별로 달라도 한 번에 처리한다."""
    out = np.zeros(len(s), bool)
    for c in range(0, len(Y), 2048):
        sl = slice(c, min(c + 2048, len(Y)))
        ph = np.exp(-2j * np.pi * (eps[sl, None] * BINHZ) * t[None, :]).astype(np.complex64)
        out[sl] = scores(Y[sl] * ph).argmax(1) == s[sl]
    return out


def ok_joint(Y, s, grid=GRID):
    best = np.full(len(Y), -np.inf); arg = np.zeros(len(Y), int)
    for f in grid:
        S = scores(Y, f); m = S.max(1); w = m > best
        best[w] = m[w]; arg[w] = S.argmax(1)[w]
    return arg == s


def ok_packet(Y, s, ns, grid=GRID):
    """패킷 단위로 CFO 를 한 번 추정한 뒤 복조"""
    out = np.zeros(len(s), bool)
    for k in range(len(Y) // ns):
        sl = slice(k * ns, (k + 1) * ns)
        Yp, sp = Y[sl], s[sl]
        bf, bs = 0.0, -np.inf
        for f in grid:
            v = scores(Yp, f).max(1).sum()
            if v > bs:
                bs, bf = v, f
        out[sl] = scores(Yp, bf).argmax(1) == sp
    return out


# =============================================================== [A] fixed CFO
head("[A] 고정 residual CFO 가 ML 검출기에 주는 손실 - 95% Wilson CI  (i.i.d. 시행)")
TA = 20000
EPS_A = [0.0, 0.10, 0.25, 0.50]
print(f"  {TA} trials/cell\n")
print(f"  {'SNR':>5} |" + "".join(f"{('eps=%.2f bin' % e):>26}" for e in EPS_A))
for snr in [-30, -25, -20, -15]:
    row = f"  {snr:>5} |"
    for e in EPS_A:
        okc = 0
        for c in range(0, TA, 2000):
            s = rng.integers(0, N, 2000)
            okc += int(ok_blind(rx(s, snr, np.full(2000, e)), s).sum())
        p, lo, hi = wilson(okc, TA)
        row += f"{p:>13.1f}% [{lo:4.1f},{hi:4.1f}]"
    print(row)

# ====================================================== [B] packet pooling vs genie
head("[B] residual CFO ~ U(-0.5,0.5) bin, 패킷 내 상수 - 95% packet-cluster bootstrap CI")
NPK, NS_BLK = 300, 128                       # 300 packets x 128 symbols = 38400 symbols/cell
print(f"  {NPK} packets x {NS_BLK} symbols = {NPK*NS_BLK} symbols/cell")
print("  CI 는 패킷을 재표집한 부트스트랩이다 (심볼은 패킷 안에서 독립이 아니다).\n")
cols = ["blind MF", "joint(sym)", "pkt ns=8", "pkt ns=32", "pkt ns=128", "genie"]
print(f"  {'SNR':>5} |" + "".join(f"{c:>23}" for c in cols) + f"{'headroom (genie-blind)':>26}")
for snr in [-30, -25, -20, -15]:
    s = rng.integers(0, N, NPK * NS_BLK)
    eps = np.repeat(np.round(rng.uniform(-0.5, 0.5, NPK), 3), NS_BLK)
    Y = rx(s, snr, eps)
    arms = {
        "blind MF": ok_blind(Y, s), "joint(sym)": ok_joint(Y, s),
        "pkt ns=8": ok_packet(Y, s, 8), "pkt ns=32": ok_packet(Y, s, 32),
        "pkt ns=128": ok_packet(Y, s, NS_BLK), "genie": ok_genie(Y, s, eps),
    }
    row = f"  {snr:>5} |"
    for c in cols:
        m, lo, hi, _ = cluster_boot(arms[c], NS_BLK, seed=1)
        row += f"{m:>11.1f}% [{lo:4.1f},{hi:4.1f}]"
    d, dlo, dhi, _ = paired_boot(arms["genie"], arms["blind MF"], NS_BLK, seed=1)
    row += f"{d:>13.1f}%p [{dlo:4.1f},{dhi:4.1f}]"
    print(row)

# =========================================================== [C] deepest SNR
head("[C] 깊은 SNR 에서도 패킷 pooling 이 genie 상한에 닿는가 - paired CI")
NPK_C, NS_C = 240, 128
print(f"  {NPK_C} packets x {NS_C} symbols = {NPK_C*NS_C} symbols/cell\n")
print(f"  {'SNR':>5} |{'blind':>22}{'pkt ns=32':>22}{'pkt ns=128':>22}{'genie':>22}"
      f"{'genie - pkt128 (paired)':>27}")
for snr in [-32, -30, -28, -26]:
    s = rng.integers(0, N, NPK_C * NS_C)
    eps = np.repeat(np.round(rng.uniform(-0.5, 0.5, NPK_C), 3), NS_C)
    Y = rx(s, snr, eps)
    ab = ok_blind(Y, s); a32 = ok_packet(Y, s, 32)
    a128 = ok_packet(Y, s, NS_C); ag = ok_genie(Y, s, eps)
    row = f"  {snr:>5} |"
    for arm in [ab, a32, a128, ag]:
        m, lo, hi, _ = cluster_boot(arm, NS_C, seed=2)
        row += f"{m:>10.1f}% [{lo:4.1f},{hi:4.1f}]"
    d, dlo, dhi, pv = paired_boot(ag, a128, NS_C, seed=2)
    row += f"{d:>13.2f}%p [{dlo:5.2f},{dhi:5.2f}]"
    print(row)

# ================================================================ [D] CFO drift
head("[D] 패킷 내 CFO drift - 상수-CFO 추정기가 blind MF 보다 나빠지는가 (핵심 주장)")
NPK_D, NS_D = 400, 128
print(f"  SNR = -25 dB, ns={NS_D}, {NPK_D} packets x {NS_D} = {NPK_D*NS_D} symbols/cell")
print("  마지막 열이 이 연구의 결론이 걸린 값이다: 대응표본 부트스트랩으로 낸")
print("  (상수-CFO 추정기 - blind MF) 의 차이와 그 95% CI, 그리고 양측 p.\n")
print(f"  {'drift(bin)':>11} |{'blind MF':>22}{'상수-CFO 추정기':>24}{'genie':>22}"
      f"{'추정기 - blind (paired)':>30}{'p':>8}")
for drift in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
    s = rng.integers(0, N, NPK_D * NS_D)
    base = np.repeat(np.round(rng.uniform(-0.5, 0.5, NPK_D), 3), NS_D)
    ramp = np.tile(np.linspace(-drift / 2, drift / 2, NS_D), NPK_D)
    eps = base + ramp
    Y = rx(s, -25, eps)
    ab = ok_blind(Y, s); ap = ok_packet(Y, s, NS_D); ag = ok_genie(Y, s, np.round(eps, 6))
    row = f"  {drift:>11.2f} |"
    for arm in [ab, ap, ag]:
        m, lo, hi, _ = cluster_boot(arm, NS_D, seed=3)
        row += f"{m:>10.1f}% [{lo:4.1f},{hi:4.1f}]"
    d, dlo, dhi, pv = paired_boot(ap, ab, NS_D, seed=3)
    row += f"{d:>15.2f}%p [{dlo:5.2f},{dhi:5.2f}]{pv:>8.4f}"
    print(row)
print("\n  genie - (추정기, blind 중 최선) 이 곧 '고전 상수-CFO 방법으로 접근 불가능한 여지'다.")
