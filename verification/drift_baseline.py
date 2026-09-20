# -*- coding: utf-8 -*-
"""
§5.4 (시변 CFO) 주장에 대한 강한 baseline 검사.

문제: "패킷 내 CFO 는 상수"라는 모델을 쓰는 추정기가 drift 에서 무너진다는 것만으로는 약하다.
      SFO/drift 는 LoRa 문헌에서 이미 알려진 문제이고 고전적 대응책도 있다 (preamble 기반
      SFO 추정, drift 보상).  리뷰어는 반드시 묻는다:
      "drift 를 모델에 넣은 고전 추정기를 쓰면 되는 것 아닌가?"

그래서 baseline 을 네 단계로 올린다:
   (1) blind MF                 - 아무것도 안 함
   (2) 상수-CFO 패킷 추정기       - 1D grid.  README §5.4 가 쓰던 baseline
   (3) drift-aware 패킷 추정기    - 2D grid (offset x slope). 선형 drift 를 정확히 모델링
   (4) genie                    - 심볼별 참 CFO 를 앎 (상한)

그리고 drift 를 두 종류로 준다:
   linear : 선형 램프.  (3) 의 모델과 **정확히 일치** -> (3) 이 상한에 닿아야 정상이다.
   walk   : 심볼 단위 random walk.  저차 파라메트릭 모델로는 안 잡힌다.
            여기 남는 여지가 진짜 '열린 조건'이다.

이 스크립트의 결과가 §5.4 / §6 의 주장을 결정한다.
"""
import sys, os, time
import numpy as np
import numpy.matlib
from scipy.signal import chirp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import cluster_boot, paired_boot

rng = np.random.default_rng(90210)
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

NS = 128
IRAMP = np.linspace(-0.5, 0.5, NS)            # 패킷 내 정규화 위치
G1 = np.arange(-0.5, 0.501, 0.125)            # 9 offset 가설
GS = np.linspace(-2.0, 2.0, 9)                # 9 slope 가설 (패킷 동안의 총 변화폭)


def ramp(eps):
    """eps: (n,) -> (n,L) 보상 위상 exp(-j2 pi eps BINHZ t)"""
    return np.exp(-2j * np.pi * (np.asarray(eps, float)[:, None] * BINHZ) * t[None, :]
                  ).astype(np.complex64)


# 가설 위상을 미리 만들어 둔다.  eps_i = f + g*IRAMP[i] 이므로 위상이
#   exp(-j2pi f BINHZ t) * exp(-j2pi g IRAMP[i] BINHZ t)
# 로 분리된다 -> (2048,) 벡터 9개와 (128,2048) 행렬 9개만 있으면 된다.
V_OFF = np.stack([ramp(np.array([f]))[0] for f in G1])            # (9, L)
V_SLP = np.stack([ramp(g * IRAMP) for g in GS])                   # (9, NS, L)
print(f"precomputed hypothesis phases: {V_OFF.nbytes/1e6:.1f} MB + {V_SLP.nbytes/1e6:.1f} MB")


def scores(Y):
    Z = np.fft.fft(Y, axis=1) * F0c[None, :]
    return np.abs(np.fft.ifft(Z.reshape(len(Y), osf, N).sum(1), axis=1))[:, LAG]


def rx(syms, snr_db, eps):
    Y = X[syms] * np.conj(ramp(eps))
    q = SIGPW / (10 ** (snr_db / 10))
    return Y + (rng.normal(0, np.sqrt(q / 2), Y.shape)
                + 1j * rng.normal(0, np.sqrt(q / 2), Y.shape)).astype(np.complex64)


def dec_blind(Yp):
    return scores(Yp).argmax(1)


def dec_const(Yp):
    bi, bs = 0, -np.inf
    for i in range(len(G1)):
        v = scores(Yp * V_OFF[i][None, :]).max(1).sum()
        if v > bs:
            bs, bi = v, i
    return scores(Yp * V_OFF[bi][None, :]).argmax(1)


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


SNR = -25
NPK = 50
ARMS = ["blind MF", "상수-CFO 추정기 (1D)", "drift-aware 추정기 (2D)", "genie"]
print(f"SNR = {SNR} dB,  ns = {NS},  {NPK} packets x {NS} = {NPK*NS} symbols per cell")
print(f"1D grid = {len(G1)} 가설,  2D grid = {len(G1)*len(GS)} 가설\n")

def rx_fixed(syms, snr_db, eps, noise):
    Y = X[syms] * np.conj(ramp(eps))
    q = SIGPW / (10 ** (snr_db / 10))
    return Y + (np.sqrt(q / 2) * noise).astype(np.complex64)

def run(kind, amp, seed=1234):
    rr = np.random.default_rng(seed)          # ← 모든 amp 가 같은 시드
    okm = {a: np.zeros(NPK * NS, bool) for a in ARMS}
    for k in range(NPK):
        s    = rr.integers(0, N, NS)
        base = float(np.round(rr.uniform(-0.5, 0.5), 3))
        u    = np.cumsum(rr.normal(0, 1/np.sqrt(NS), NS)); u -= u.mean()   # 단위 walk
        nz   = (rr.normal(0, 1, (NS, L)) + 1j*rr.normal(0, 1, (NS, L)))
        eps  = base + (amp * IRAMP if kind == "linear" else amp * u)       # amp 는 스케일만
        Yp   = rx_fixed(s, SNR, eps, nz)
        sl   = slice(k*NS, (k+1)*NS)
        okm["blind MF"][sl]              = dec_blind(Yp) == s
        okm["상수-CFO 추정기 (1D)"][sl]    = dec_const(Yp) == s
        okm["drift-aware 추정기 (2D)"][sl] = dec_drift(Yp) == s
        okm["genie"][sl]                 = dec_genie(Yp, eps) == s
    return okm


for kind in ["linear", "walk"]:
    print("=" * 112)
    print(f"[{kind}] 패킷 내 CFO 변화  ({'선형 램프' if kind == 'linear' else '심볼 단위 random walk'})")
    print("=" * 112)
    print(f"  {'변화폭(bin)':>11} |" + "".join(f"{a:>24}" for a in ARMS)
          + f"{'genie - 2D (paired)':>26}{'p':>8}")
    for amp in [0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]:
        t0 = time.perf_counter()
        okm = run(kind, amp)
        row = f"  {amp:>11.2f} |"
        for a in ARMS:
            m, lo, hi, _ = cluster_boot(okm[a], NS, seed=7)
            row += f"{m:>12.1f}% [{lo:4.1f},{hi:4.1f}]"
        d, dlo, dhi, pv = paired_boot(okm["genie"], okm["drift-aware 추정기 (2D)"], NS, seed=7)
        row += f"{d:>11.2f}%p [{dlo:5.2f},{dhi:5.2f}]{pv:>8.4f}"
        print(row)
        d2, d2lo, d2hi, pv2 = paired_boot(okm["drift-aware 추정기 (2D)"],
                                          okm["상수-CFO 추정기 (1D)"], NS, seed=7)
        d3, d3lo, d3hi, pv3 = paired_boot(okm["상수-CFO 추정기 (1D)"], okm["blind MF"], NS, seed=7)
        print(f"  {'':>11}   2D - 1D = {d2:+6.2f}%p [{d2lo:6.2f},{d2hi:6.2f}] p={pv2:.4f}"
              f"   |   1D - blind = {d3:+6.2f}%p [{d3lo:6.2f},{d3hi:6.2f}] p={pv3:.4f}"
              f"   ({time.perf_counter()-t0:.0f}s)")
    print()