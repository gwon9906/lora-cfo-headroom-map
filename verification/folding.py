# -*- coding: utf-8 -*-
"""
Q (reviewer): OSF=4 에서 dechirp+FFT 가 심볼 에너지를 두 alias 조각으로 쪼갠다면,
   그 두 조각을 **복소(coherent) 합**으로 합치면 최적 검출기가 되는 것 아닌가?
   그렇다면 최적 검출기는 FFT 2회가 아니라 512-FFT 1회로 끝나는 것 아닌가?

이 스크립트는 그 질문에만 답한다.  구성:
  [0] 구현 정합성 검사 (크기합이 아니라 복소합인지, 그리고
      주파수영역 512-블록 복소합 == 시간영역 4-데시메이션 후 512-FFT 인지)
  [1] dechirp 된 템플릿의 구조 — 조각이 왜 둘인지, 경계가 어디인지
  [2] 모든 선형 결합기의 출력 SNR 손실을 해석적으로 계산 (몬테카를로 불필요)
  [3] 동일 잡음 실현 몬테카를로 — 정확도 + 95% Wilson CI + direct MF 와의 판정 일치율
  [4] top-K bin MRC: 몇 개의 bin 을 모아야 ML 에 닿는가
  [5] 연산량
"""
import sys, os, time
import numpy as np
import numpy.matlib
from scipy.signal import chirp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import wilson

rng = np.random.default_rng(20260915)
sf, bw, OSF = 9, 250_000, 4
fs = bw * OSF
N = 2 ** sf


def gen(cw):
    t = np.arange(0, 2 ** sf / bw, 1 / fs); nsm = fs * 2 ** sf / bw
    ci = chirp(t, -bw / 2, 2 ** sf / bw, bw / 2, 'linear', 0)
    cq = chirp(t, -bw / 2, 2 ** sf / bw, bw / 2, 'linear', -90)
    b = numpy.matlib.repmat(ci + 1j * cq, 1, 2); off = round((2 ** sf - cw) / 2 ** sf * nsm)
    return b[:, int(nsm - off):int(nsm - off + int(nsm))][0]


X = np.array([gen(m) for m in range(N)])              # (512, 2048) complex128
L = X.shape[1]
osf = L // N
P = X / np.linalg.norm(X, axis=1, keepdims=True)
tt = np.arange(L) / fs
Ts = N / bw
DC = np.conj(np.exp(1j * 2 * np.pi * (-bw / 2 * tt + (bw / (2 * Ts)) * tt ** 2)))

# clean dechirped templates  c_m[n] = X[m][n] * DC[n]   and their spectra
C = np.fft.fft(X * DC[None, :], axis=1)               # (512, 2048)
SIGBLK = (0, 3)                                       # blocks that actually carry signal

BAR = "=" * 78


def head(s):
    print("\n" + BAR + "\n" + s + "\n" + BAR)


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-300))


# ---------------------------------------------------------------- [0] identities
head("[0] 구현 정합성 - 복소합인가, 그리고 데시메이션과 같은가")

y_t = X[137] + (rng.normal(0, .3, L) + 1j * rng.normal(0, .3, L))
yd = y_t * DC
F = np.fft.fft(yd)

fold_freq = F.reshape(osf, N).sum(0)                  # 주파수영역 512-블록 **복소**합
fold_time = osf * np.fft.fft(yd[::osf])               # 시간영역 4-데시메이션 후 512-FFT
e0 = np.abs(fold_freq - fold_time).max() / np.abs(fold_time).max()
print("  |4-block complex fold  -  4*FFT512(dechirped[::4])| / max = "
      "{:.3e}   -> {}".format(e0, "IDENTICAL" if e0 < 1e-10 else "MISMATCH"))
print("     (두 구현이 같다 == 복소합을 제대로 한 것. 크기합이면 여기서 깨진다.)")

fold_mag = (np.abs(F).reshape(osf, N)).sum(0)         # 대조군: 크기합
print("  참고) 크기합 vs 복소합 상관 = {:.4f}   (둘은 다른 통계량이다)".format(
    np.corrcoef(fold_mag, np.abs(fold_freq))[0, 1]))

F0c = np.conj(np.fft.fft(X[0]))
IDX = (-osf * np.arange(N)) % L
mf_full = np.fft.ifft(np.fft.fft(y_t) * F0c)[IDX]
prod = np.fft.fft(y_t) * F0c
mf_dec = (np.fft.ifft(prod.reshape(osf, N).sum(0)) / osf)[(-np.arange(N)) % N]  # lag=4m -> 512-IFFT
e1 = np.abs(mf_full - mf_dec).max() / np.abs(mf_full).max()
print("  |MF via 2048-IFFT  -  MF via folded 512-IFFT| / max = "
      "{:.3e}   -> {}".format(e1, "IDENTICAL" if e1 < 1e-10 else "MISMATCH"))
print("     (=> 최적 검출기는 2048-FFT 1회 + 512-IFFT 1회. 2048-FFT 2회가 아니다.)")

assert all(np.allclose(X[m], np.roll(X[0], -osf * m), atol=1e-6) for m in [1, 7, 100, 511])
print("  X[m] == roll(X[0], -4m)  for all tested m  -> OK")

# ---------------------------------------------------------------- [1] structure
head("[1] dechirp 된 심볼은 왜 '두 조각'인가")
print("  c_m[n] = X[m][n]*conj(upchirp[n]) 은 두 개의 시간 구간이다:")
print("    n <  n0 : bin m      의 톤        (n0 = L - OSF*m, cyclic shift 의 wrap 위치)")
print("    n >= n0 : bin m-512  의 톤  = bin m 톤 x (-j)^n  (주파수가 정확히 -BW 만큼 점프)")
print("  두 조각은 '같은 톤의 복제본 두 개'가 아니라 **겹치지 않는 두 시간 구간**이다.\n")
print("  {:>4} {:>9} {:>13} {:>18} {:>9} {:>19}".format(
    "m", "n0=L-4m", "|C[m]|^2/L^2", "|C[m+1536]|^2/L^2", "2-bin 합", "예측 (n0^2+(L-n0)^2)/L^2"))
for m in [0, 64, 100, 256, 300, 400, 511]:
    a = np.abs(C[m, m]) ** 2 / L ** 2
    b = np.abs(C[m, (m - N) % L]) ** 2 / L ** 2
    n0 = L - osf * m
    pred = (n0 ** 2 + (L - n0) ** 2) / L ** 2
    print("  {:>4} {:>9} {:>13.4f} {:>18.4f} {:>9.4f} {:>19.4f}".format(m, n0, a, b, a + b, pred))
print("  -> 두 bin 이 담는 에너지 비율은 심볼 인덱스에 따라 100% ~ 50% 로 변한다.")
print("     (m=0 은 wrap 이 없어 한 조각, m=256 은 반반. 손실이 심볼마다 다르다.)")

# ---------------------------------------------------------------- [2] analytic loss
head("[2] 각 결합기의 출력 SNR 손실 (해석적, 몬테카를로 아님)")
print("  선형 결합기 g 의 출력 SNR = |<c_m, g>|^2 / (||g||^2 sigma^2).")
print("  ML(=MF) 대비 손실 = |<c_m,g>|^2 / (||g||^2 ||c_m||^2).  결정론적 값이다.\n")

k0 = np.arange(N)
k3 = (np.arange(N) - N) % L
C0, C3 = C[k0, k0], C[k0, k3]

loss = {}
loss["bin m 하나만 (pre-fix power[:512])"] = np.abs(C0) ** 2 / L ** 2
loss["2-bin 등이득 복소합 (EGC)"] = (np.abs(np.abs(C0) + np.abs(C3)) ** 2) / (2 * L * L)
loss["2-bin MRC (최적 2-bin 결합)"] = (np.abs(C0) ** 2 + np.abs(C3) ** 2) / L ** 2
Csum = C[k0[:, None], (k0[:, None] + N * np.arange(osf)[None, :]) % L].sum(1)
loss["4-block 복소합 (= 4-데시메이션)"] = np.abs(Csum) ** 2 / (osf * L * L)
loss["전 bin MRC (= matched filter)"] = (np.abs(C) ** 2).sum(1) / L ** 2

print("  {:<36}{:>11}{:>11}{:>11}".format("결합기", "평균 손실", "최선", "최악"))
for k, v in loss.items():
    print("  {:<36}{:>9.2f}dB{:>9.2f}dB{:>9.2f}dB".format(k, db(v.mean()), db(v.max()), db(v.min())))
print("\n  핵심: 4-block 복소합은 **모든 심볼에서 정확히 6.02 dB** 손실이다.")
print("  이유: 주파수영역 4-블록 복소합 == 시간영역 4-데시메이션.  대역제한 없이 4:1 로")
print("  데시메이트하면 1 MHz 에 퍼져 있던 잡음이 250 kHz 로 4배 접혀 들어온다.")
print("  신호는 그대로, 잡음만 4배 -> 6 dB.  '복소로 합쳤는데 왜 더 나빠지나'의 답.")

# ---------------------------------------------------------------- detectors
sl_lo, sl_hi = slice(0, N // 2), slice(L - N // 2, L)
t512 = np.arange(N) / bw
DC512 = np.conj(np.exp(1j * 2 * np.pi * (-bw / 2 * t512 + (bw / (2 * Ts)) * t512 ** 2)))
_Z0 = np.fft.fft(X[0])
X0dec = np.fft.ifft(np.concatenate([_Z0[sl_lo], _Z0[sl_hi]]))
F0dec = np.conj(np.fft.fft(X0dec))
Pc = P.conj().T.astype(np.complex64)
PH0, PH3 = np.exp(-1j * np.angle(C0)), np.exp(-1j * np.angle(C3))


def decimate_bl(Y):
    """이상적 대역제한(|f| < BW/2) 후 4:1 데시메이션 -> 250 kHz, 512 샘플"""
    Z = np.fft.fft(Y, axis=1)
    return np.fft.ifft(np.concatenate([Z[:, sl_lo], Z[:, sl_hi]], axis=1), axis=1)


def detectors(Y):
    """Y:(n,L) -> {name: decisions}"""
    Fd = np.fft.fft(Y * DC[None, :], axis=1)
    B = Fd.reshape(len(Y), osf, N)
    Pw = np.abs(B) ** 2
    d = {}
    d["pre-fix  power[:512]"] = Pw[:, 0, :].argmax(1)
    d["repo  4-block |.|^2 합"] = Pw.sum(1).argmax(1)
    d["신호블록 2개 |.|^2 합"] = (Pw[:, SIGBLK[0], :] + Pw[:, SIGBLK[1], :]).argmax(1)
    d["신호블록 2개 |.| 합 (크기)"] = (np.abs(B[:, SIGBLK[0], :]) + np.abs(B[:, SIGBLK[1], :])).argmax(1)
    d["4-block 복소합"] = np.abs(B.sum(1)).argmax(1)
    d["4-데시메이션 + 512FFT"] = np.abs(np.fft.fft((Y * DC[None, :])[:, ::osf], axis=1)).argmax(1)
    d["2-bin 등이득 복소합"] = np.abs(B[:, SIGBLK[0], :] * PH0[None, :]
                                  + B[:, SIGBLK[1], :] * PH3[None, :]).argmax(1)
    d["2-bin MRC (최적 2-bin)"] = np.abs(B[:, SIGBLK[0], :] * np.conj(C0)[None, :]
                                        + B[:, SIGBLK[1], :] * np.conj(C3)[None, :]).argmax(1)
    Yd = decimate_bl(Y)
    d["대역제한+데시메이션, 512FFT"] = np.abs(np.fft.fft(Yd * DC512[None, :], axis=1)).argmax(1)
    d["대역제한+데시메이션, 512MF"] = np.abs(
        np.fft.ifft(np.fft.fft(Yd, axis=1) * F0dec[None, :], axis=1))[:, (-np.arange(N)) % N].argmax(1)
    d["FFT-MF (2048FFT+512IFFT)"] = (-np.abs(np.fft.ifft(
        (np.fft.fft(Y, axis=1) * F0c[None, :]).reshape(len(Y), osf, N).sum(1),
        axis=1)).argmax(1)) % N
    d["direct MF bank (기준 ML)"] = np.abs(Y.astype(np.complex64) @ Pc).argmax(1)
    return d


ORDER = list(detectors(X[:1] + 0j).keys())

# ---------------------------------------------------------------- [3] monte carlo
head("[3] 동일 잡음 실현 몬테카를로 - 정확도 [95% Wilson CI] / direct MF 판정 일치율")
T, CH = 6000, 500
SNRS = [-30, -25, -20, -15]
res = {k: {s: 0 for s in SNRS} for k in ORDER}
agree = {k: {s: 0 for s in SNRS} for k in ORDER}
pw = np.mean(np.abs(X[0]) ** 2)
for snr in SNRS:
    q = pw / (10 ** (snr / 10))
    for c0 in range(0, T, CH):
        n = min(CH, T - c0)
        s = rng.integers(0, N, n)
        Y = X[s] + (rng.normal(0, np.sqrt(q / 2), (n, L))
                    + 1j * rng.normal(0, np.sqrt(q / 2), (n, L)))
        d = detectors(Y)
        ref = d["direct MF bank (기준 ML)"]
        for k, v in d.items():
            res[k][snr] += int((v == s).sum())
            agree[k][snr] += int((v == ref).sum())

print("  {} trials/SNR, SF9 / BW 250k / OSF 4\n".format(T))
print("  {:<30}".format("detector") + "".join("{:>24}".format(str(s) + " dB") for s in SNRS)
      + "{:>10}".format("MF 일치"))
for k in ORDER:
    row = "  {:<30}".format(k)
    for s in SNRS:
        p, lo, hi = wilson(res[k][s], T)
        row += "{:>10.1f}% [{:4.1f},{:4.1f}]".format(p, lo, hi)
    row += "{:>9.1f}%".format(np.mean([agree[k][s] for s in SNRS]) / T * 100)
    print(row)

# ------------------------------------------------------- [3b] equivalent SNR loss
head("[3b] 정확도를 등가 SNR 손실(dB)로 환산 - ML 곡선에 역투영")
GS = np.arange(-34.0, -13.9, 0.5)
curve = []
for snr in GS:
    q = pw / (10 ** (snr / 10)); ok = 0; TT = 4000
    for c0 in range(0, TT, 1000):
        s = rng.integers(0, N, 1000)
        Y = X[s] + (rng.normal(0, np.sqrt(q / 2), (1000, L))
                    + 1j * rng.normal(0, np.sqrt(q / 2), (1000, L)))
        dec = (-np.abs(np.fft.ifft(
            (np.fft.fft(Y, axis=1) * F0c[None, :]).reshape(1000, osf, N).sum(1), axis=1)
        ).argmax(1)) % N
        ok += int((dec == s).sum())
    curve.append(ok / TT)
curve = np.array(curve)
mono = np.maximum.accumulate(curve)


def eq_snr(p):
    if p <= mono[0] or p >= mono[-1]:
        return np.nan
    return float(np.interp(p, mono, GS))


print("  ML 정확도 곡선을 0.5 dB 격자로 만든 뒤, 각 검출기의 정확도를 그 곡선에 역투영해")
print("  '같은 정확도를 내려면 ML 은 몇 dB 에서 돌면 되는가'를 구한다. 차이가 실질 손실이다.")
print()
print("  {:<30}".format("detector") + "".join("{:>12}".format(str(sv) + " dB") for sv in SNRS)
      + "{:>10}".format("평균"))
for k in ORDER:
    vals = []
    row = "  {:<30}".format(k)
    for sv in SNRS:
        e = eq_snr(res[k][sv] / T)
        if np.isnan(e):
            row += "{:>12}".format("-")
        else:
            row += "{:>11.2f}".format(e - sv)
            vals.append(e - sv)
    row += "{:>10}".format("{:.2f} dB".format(np.mean(vals)) if vals else "-")
    print(row)
print()
print("  ([2] 의 해석적 예측과 비교: 4-block 복소합 -6.02 dB, 2-bin MRC -1.76 dB 평균)")

# ---------------------------------------------------------------- [4] top-K MRC
head("[4] top-K bin MRC - ML 에 닿으려면 bin 이 몇 개 필요한가")
ordr = np.argsort(np.abs(C) ** 2, axis=1)[:, ::-1]
print("  {:>6}{:>18}{:>16}{:>18}".format("K", "평균 포착 에너지", "평균 SNR 손실", "곱셈/심볼"))
for K in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    idx = ordr[:, :K]
    frac = (np.abs(C[np.arange(N)[:, None], idx]) ** 2).sum(1) / L ** 2
    print("  {:>6}{:>17.2f}%{:>14.2f}dB{:>18,}".format(K, frac.mean() * 100, db(frac.mean()), 512 * K))
print("  -> post-FFT bin 결합으로 ML 에 닿으려면 사실상 전 bin 을 심볼별 복소 가중치로")
print("     모아야 하고, 그것은 정의상 matched filter 다. bin 2개로는 닫히지 않는다.")

# ---------------------------------------------------------------- [5] cost
head("[5] 심볼당 연산 시간")
y1 = (X[3] + 0j)[None, :]
cases = [
    ("repo dechirp+2048FFT",
     lambda: np.abs(np.fft.fft(y1 * DC[None, :], axis=1)).reshape(1, osf, N).sum(1).argmax()),
    ("4-데시메이션+512FFT",
     lambda: np.abs(np.fft.fft((y1 * DC[None, :])[:, ::osf], axis=1)).argmax()),
    ("대역제한+데시메이션+512FFT",
     lambda: np.abs(np.fft.fft(decimate_bl(y1) * DC512[None, :], axis=1)).argmax()),
    ("FFT-MF (2048FFT+512IFFT)",
     lambda: (-np.abs(np.fft.ifft((np.fft.fft(y1, axis=1) * F0c[None, :]).reshape(1, osf, N).sum(1),
                                 axis=1)).argmax()) % N),
    ("FFT-MF (2048FFT x2, 구버전)",
     lambda: np.abs(np.fft.ifft(np.fft.fft(y1, axis=1) * F0c[None, :], axis=1)[:, IDX]).argmax()),
    ("direct MF bank 512x2048",
     lambda: np.abs(y1.astype(np.complex64) @ Pc).argmax()),
]
for name, f in cases:
    for _ in range(50):
        f()
    t0 = time.perf_counter()
    for _ in range(2000):
        f()
    print("  {:<30}{:>9.1f} us".format(name, (time.perf_counter() - t0) / 2000 * 1e6))
print("\n  주: 대역제한 데시메이션을 여기서는 FFT 로 구현해 2048-FFT 비용이 들어간다.")
print("     실제 수신기는 ADC 직후 polyphase FIR 로 250 kHz 까지 내려받으므로, 그 경우")
print("     심볼 복조는 512-FFT 1회로 끝나고 그것이 ML 이다 ([3] 의 판정 일치율 참조).")
