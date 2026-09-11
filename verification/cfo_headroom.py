"""
CFO headroom test.
Q: AWGN-only 에서는 MF(=ML) 가 최적이라 front-end 로 얻을 것이 없다.
   그렇다면 residual CFO 가 있을 때는 '학습 기반 복조'가 노릴 여지가 실제로 있는가?
   있다면 몇 %p 인가? 그리고 그 여지를 고전적 방법(joint grid search)이 이미
   다 먹어버리는가?
"""
import numpy as np, numpy.matlib
from scipy.signal import chirp

rng = np.random.default_rng(7)
sf, bw, OSF = 9, 250_000, 4
fs = bw * OSF
N = 2**sf
BINHZ = bw / N                      # 488.28 Hz per symbol bin

def gen(cw):
    t = np.arange(0, 2**sf/bw, 1/fs); nsm = fs*2**sf/bw
    ci = chirp(t, -bw/2, 2**sf/bw, bw/2, 'linear', 0)
    cq = chirp(t, -bw/2, 2**sf/bw, bw/2, 'linear', -90)
    b = numpy.matlib.repmat(ci+1j*cq, 1, 2); off = round((2**sf-cw)/2**sf*nsm)
    return b[:, int(nsm-off):int(nsm-off+int(nsm))][0]

X = np.array([gen(m) for m in range(N)], dtype=np.complex64)
L = X.shape[1]
P = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.complex64)   # (512, 2048)
t = (np.arange(L)/fs).astype(np.float64)

def rx(syms, snr_db, eps_bins):
    """syms:(T,) eps_bins:(T,) fractional CFO in bins. returns (T,L) complex64"""
    Y = X[syms].copy()
    ph = np.exp(2j*np.pi*(eps_bins[:, None]*BINHZ)*t[None, :]).astype(np.complex64)
    Y = Y*ph
    p = np.mean(np.abs(X[0])**2)
    q = p/(10**(snr_db/10))
    n = (rng.normal(0, np.sqrt(q/2), Y.shape)+1j*rng.normal(0, np.sqrt(q/2), Y.shape)).astype(np.complex64)
    return Y+n

def mf_scores(Y, f_bins=0.0):
    """correlate Y against prototypes pre-rotated by f_bins -> (T,512) magnitudes"""
    Pf = P*np.exp(2j*np.pi*(f_bins*BINHZ)*t[None, :]).astype(np.complex64) if f_bins != 0.0 else P
    return np.abs(Y @ Pf.conj().T)

def acc_blind(Y, syms):
    return (mf_scores(Y).argmax(1) == syms).mean()*100

def acc_genie(Y, syms, eps):
    ok = 0
    for u in np.unique(eps):
        i = np.where(eps == u)[0]
        ok += (mf_scores(Y[i], u).argmax(1) == syms[i]).sum()
    return ok/len(syms)*100

def acc_joint(Y, syms, grid):
    """per-symbol joint (symbol, CFO) search over grid"""
    best = np.full(len(Y), -np.inf); arg = np.zeros(len(Y), int)
    for f in grid:
        S = mf_scores(Y, f)
        m = S.max(1); w = m > best
        best[w] = m[w]; arg[w] = S.argmax(1)[w]
    return (arg == syms).mean()*100

def acc_packet(Y, syms, grid, ns):
    """CFO constant over a packet of ns symbols -> estimate CFO jointly, then decode"""
    T = len(Y); npk = T//ns; ok = 0
    for k in range(npk):
        sl = slice(k*ns, (k+1)*ns)
        Yp, sp = Y[sl], syms[sl]
        bf, bs = None, -np.inf
        for f in grid:
            s = mf_scores(Yp, f).max(1).sum()      # packet-level CFO likelihood
            if s > bs: bs, bf = s, f
        ok += (mf_scores(Yp, bf).argmax(1) == sp).sum()
    return ok/(npk*ns)*100

SNRS = [-30, -25, -20, -15]
T = 1200
GRID = np.arange(-0.5, 0.501, 0.125)      # 9 hypotheses, +-0.5 bin
NS = 8

print("SF9 / BW250k / OSF4 / matched-filter(=ML) detector, %d trials per cell\n" % T)
print("[A] 고정 residual CFO 가 MF 정확도에 주는 손실  (CFO=0 이 AWGN 최적 상한)")
hdr = f"{'SNR':>5} |" + "".join(f"{('e=%.2f bin'%e):>13}" for e in [0.0, 0.10, 0.25, 0.50])
print(hdr); print("-"*len(hdr))
for snr in SNRS:
    row = f"{snr:>5} |"
    for e in [0.0, 0.10, 0.25, 0.50]:
        s = rng.integers(0, N, T)
        Y = rx(s, snr, np.full(T, e))
        row += f"{acc_blind(Y, s):>12.1f}%"
    print(row)

print("\n[B] residual CFO ~ U(-0.5,0.5) bin  :  blind MF / genie(CFO known) / per-symbol joint / packet-joint(ns=8)")
hdr = f"{'SNR':>5} |{'blind MF':>11}{'genie':>11}{'joint(sym)':>13}{'joint(pkt)':>13}  {'headroom':>10}"
print(hdr); print("-"*len(hdr))
for snr in SNRS:
    s = rng.integers(0, N, T)
    eps = np.round(rng.uniform(-0.5, 0.5, T), 3)
    Y = rx(s, snr, eps)
    ab = acc_blind(Y, s); ag = acc_genie(Y, s, eps)
    aj = acc_joint(Y, s, GRID); ap = acc_packet(Y, s, GRID, NS)
    print(f"{snr:>5} |{ab:>10.1f}%{ag:>10.1f}%{aj:>12.1f}%{ap:>12.1f}%  {ag-ab:>9.1f}%p")
