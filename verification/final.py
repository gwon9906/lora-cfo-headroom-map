import numpy as np
from scipy.signal import chirp
import numpy.matlib
rng=np.random.default_rng(42)
sf,bw,OSF=9,250_000,4; fs=bw*OSF; N=2**sf; ns=N
def gen(cw):
    t=np.arange(0,2**sf/bw,1/fs); nsm=fs*2**sf/bw
    ci=chirp(t,-bw/2,2**sf/bw,bw/2,'linear',0); cq=chirp(t,-bw/2,2**sf/bw,bw/2,'linear',-90)
    b=numpy.matlib.repmat(ci+1j*cq,1,2); off=round((2**sf-cw)/2**sf*nsm)
    return b[:,int(nsm-off):int(nsm-off+int(nsm))][0]
X=np.array([gen(m) for m in range(N)]); P=X/np.linalg.norm(X,axis=1,keepdims=True)
L=X.shape[1]; osf=L//ns
t=np.arange(L)/fs; Ts=ns/bw
DC=np.conj(np.exp(1j*2*np.pi*(-bw/2*t+(bw/(2*Ts))*t**2)))

def repo(y):     # exactly utils/my_lora_utils.estimate_symbol_custom
    p=np.abs(np.fft.fft(y*DC))**2
    return int(np.argmax(p.reshape(osf,ns).sum(0)))
def repo_old(y): # the pre-fix version described in README appendix
    p=np.abs(np.fft.fft(y*DC))**2
    return int(np.argmax(p[:ns]))
def mf(y):       # textbook noncoherent matched filter over the 512 orthogonal prototypes
    return int(np.argmax(np.abs(P.conj()@y)))

T=3000
print("Identical noise realisations, 3000 trials/SNR, SF9 BW250k OSF4\n")
print(f"{'SNR':>5} {'README-appendix (power[:512])':>30} {'repo OSF-fold':>15} {'matched filter':>16}")
for snr in [-30,-25,-20,-15]:
    syms=rng.integers(0,N,T); a=b=c=0
    for s_ in syms:
        x=X[s_]; q=np.mean(np.abs(x)**2)/(10**(snr/10))
        y=x+(rng.normal(0,np.sqrt(q/2),L)+1j*rng.normal(0,np.sqrt(q/2),L))
        a+=repo_old(y)==s_; b+=repo(y)==s_; c+=mf(y)==s_
    print(f"{snr:>5} {a/T*100:>29.1f}% {b/T*100:>14.1f}% {c/T*100:>15.1f}%")
print("\nREADME reported: baseline -30dB=3.5% -25dB=18.7% ; BAMv3 -30dB=7.1% -25dB=27.8%")
