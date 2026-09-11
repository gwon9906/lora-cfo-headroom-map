import numpy as np
from scipy.signal import chirp, resample_poly
import numpy.matlib
rng = np.random.default_rng(1)
sf, bw, OSF = 9, 250_000, 4
fs = bw*OSF; N = 2**sf

def gen_symbol_fs(cw, sf, bw, Fs):
    o=Fs
    if Fs<bw: Fs=bw
    t=np.arange(0,2**sf/bw,1/Fs); ns=Fs*2**sf/bw
    ci=chirp(t,-bw/2,2**sf/bw,bw/2,'linear',0); cq=chirp(t,-bw/2,2**sf/bw,bw/2,'linear',-90)
    b=numpy.matlib.repmat(ci+1j*cq,1,2); off=round((2**sf-cw)/2**sf*ns)
    s=b[:,int(ns-off):int(ns-off+int(ns))]
    if o!=Fs: s=s[:,::int(Fs/o)]
    return s[0]

X = np.array([gen_symbol_fs(m,sf,bw,fs) for m in range(N)])

def awgn_iq(sig,snr):
    p=np.mean(np.abs(sig)**2); npw=p/(10**(snr/10))
    return sig+(rng.normal(0,np.sqrt(npw/2),sig.shape)+1j*rng.normal(0,np.sqrt(npw/2),sig.shape))

# --- repo's detector (OSF-fold over ALL osf blocks) ---
def est_repo(a, sf, fs, bw):
    ns=2**sf; L=len(a); L=(L//ns)*ns; a=a[:L]; osf=L//ns
    t=np.arange(L)/fs; Ts=ns/bw
    dc=np.conj(np.exp(1j*2*np.pi*(-bw/2*t+(bw/(2*Ts))*t**2)))
    p=np.abs(np.fft.fft(a*dc))**2
    return int(np.argmax(p.reshape(osf,ns).sum(0))) if osf>1 else int(np.argmax(p[:ns]))

# --- matched-filter / ML detector: decimate to BW then 512-pt FFT ---
def est_ml(a, sf, fs, bw):
    ns=2**sf; L=len(a); osf=L//ns
    t=np.arange(L)/fs; Ts=ns/bw
    dc=np.conj(np.exp(1j*2*np.pi*(-bw/2*t+(bw/(2*Ts))*t**2)))
    d=a*dc
    d=d.reshape(-1,osf).sum(1) if False else resample_poly(d,1,osf)   # LPF+decimate to BW
    return int(np.argmax(np.abs(np.fft.fft(d,ns))**2))

TRIALS=2000
print(f"{'SNR(dB)':>8} {'in-band SNR':>12} {'repo detector':>14} {'ML detector':>13} {'chance':>8}")
for snr in [-30,-25,-20,-15,-10,-5]:
    syms = rng.integers(0,N,TRIALS)
    r=m=0
    for s_ in syms:
        y=awgn_iq(X[s_],snr)
        r += est_repo(y,sf,fs,bw)==s_
        m += est_ml(y,sf,fs,bw)==s_
    print(f"{snr:>8} {snr+10*np.log10(OSF):>11.1f}  {r/TRIALS*100:>13.1f}% {m/TRIALS*100:>12.1f}% {100/N:>7.2f}%")
