# -*- coding: utf-8 -*-
"""
§5.3 의 교란변수 제거.

[C] 에서 깊은 SNR(-32 ~ -26 dB)의 패킷 CFO 추정기가 genie 상한에 못 닿는 것이 확인됐다.
그런데 그 격차에는 두 원인이 섞여 있다:

  (i)  추정기가 저 SNR 에서 CFO 를 못 맞춘다        <- 진짜 '열린 여지'
  (ii) grid 가 0.125 bin 간격이라 맞춰도 잔차가 남는다 <- 단순 구현 선택, 여지가 아님

둘을 분리하지 않으면 "고전 방법이 실패한다"는 주장이 성립하지 않는다.
grid 간격을 0.125 -> 0.03125 로 4배 촘촘하게 해서 (ii) 를 거의 제거한 뒤 다시 잰다.
격차가 그대로면 (i), 사라지면 (ii) 였던 것이다.
"""
import sys, os
import numpy as np
import numpy.matlib
from scipy.signal import chirp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import cluster_boot, paired_boot

rng = np.random.default_rng(555)
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


def ramp(eps):
    return np.exp(-2j * np.pi * (np.asarray(eps, float)[:, None] * BINHZ) * t[None, :]
                  ).astype(np.complex64)


def scores(Y):
    Z = np.fft.fft(Y, axis=1) * F0c[None, :]
    return np.abs(np.fft.ifft(Z.reshape(len(Y), osf, N).sum(1), axis=1))[:, LAG]


def rx(syms, snr_db, eps):
    Y = X[syms] * np.conj(ramp(eps))
    q = SIGPW / (10 ** (snr_db / 10))
    return Y + (rng.normal(0, np.sqrt(q / 2), Y.shape)
                + 1j * rng.normal(0, np.sqrt(q / 2), Y.shape)).astype(np.complex64)


GRIDS = {
    "coarse (0.125 bin, 9 가설)": np.arange(-0.5, 0.501, 0.125),
    "fine   (0.03125 bin, 33 가설)": np.arange(-0.5, 0.5001, 0.03125),
}
VEC = {k: np.stack([ramp(np.array([f]))[0] for f in g]) for k, g in GRIDS.items()}

NPK = 300
print(f"ns = {NS}, {NPK} packets x {NS} = {NPK*NS} symbols/cell, CFO ~ U(-0.5,0.5) 패킷 내 상수")
print("CI 는 packet-cluster bootstrap. 마지막 두 열이 핵심이다.\n")
print(f"  {'SNR':>5} |{'blind':>20}" + "".join(f"{('pkt, ' + k.split(' ')[0]):>22}" for k in GRIDS)
      + f"{'genie':>20}{'genie - fine (paired)':>26}")
for snr in [-32, -30, -28, -26]:
    s = rng.integers(0, N, NPK * NS)
    eps = np.repeat(np.round(rng.uniform(-0.5, 0.5, NPK), 4), NS)
    Y = rx(s, snr, eps)
    arms = {}
    arms["blind"] = scores(Y).argmax(1) == s
    for name, V in VEC.items():
        out = np.zeros(len(s), bool)
        for k in range(NPK):
            sl = slice(k * NS, (k + 1) * NS)
            Yp, sp = Y[sl], s[sl]
            bi, bs = 0, -np.inf
            for i in range(len(V)):
                v = scores(Yp * V[i][None, :]).max(1).sum()
                if v > bs:
                    bs, bi = v, i
            out[sl] = scores(Yp * V[bi][None, :]).argmax(1) == sp
        arms[name] = out
    gen_ok = np.zeros(len(s), bool)
    for c in range(0, len(Y), 2048):
        sl = slice(c, min(c + 2048, len(Y)))
        gen_ok[sl] = scores(Y[sl] * ramp(eps[sl])).argmax(1) == s[sl]
    arms["genie"] = gen_ok

    row = f"  {snr:>5} |"
    for key in ["blind"] + list(GRIDS) + ["genie"]:
        m, lo, hi, _ = cluster_boot(arms[key], NS, seed=11)
        row += f"{m:>9.1f}% [{lo:4.1f},{hi:4.1f}]" if key in ("blind", "genie") \
            else f"{m:>11.1f}% [{lo:4.1f},{hi:4.1f}]"
    fine = list(GRIDS)[1]
    d, lo, hi, pv = paired_boot(arms["genie"], arms[fine], NS, seed=11)
    row += f"{d:>12.2f}%p [{lo:5.2f},{hi:5.2f}]"
    print(row)
    dq, qlo, qhi, qp = paired_boot(arms[fine], arms[list(GRIDS)[0]], NS, seed=11)
    print(f"  {'':>5}   fine - coarse = {dq:+.2f}%p [{qlo:.2f},{qhi:.2f}] p={qp:.4f}"
          f"   (이 값이 grid 양자화 몫, 위 열이 남는 추정 실패 몫)")
