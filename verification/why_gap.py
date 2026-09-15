import io, numpy as np
exec(io.open('cfo_headroom.py', encoding='utf-8').read().split('def rx(')[0])
L=X.shape[1]; ns=N; osf=L//ns
tt=np.arange(L)/fs; Ts=ns/bw
DC=np.conj(np.exp(1j*2*np.pi*(-bw/2*tt+(bw/(2*Ts))*tt**2)))

# where does a clean dechirped symbol actually put its energy in the 2048-pt FFT?
for m in [0, 100, 300, 500]:
    p=np.abs(np.fft.fft(X[m]*DC))**2
    top=np.argsort(p)[::-1][:4]
    print(f"sym {m:>3}: top FFT bins {sorted(top.tolist())}  "
          f"energy in those 4 bins = {p[top].sum()/p.sum()*100:.1f}%   "
          f"blocks(bin//512) = {sorted(set((top//512).tolist()))}")

print()
def run(det,T=3000,seed=5):
    r=np.random.default_rng(seed); out={}
    for snr in [-30,-25,-20,-15]:
        s=r.integers(0,N,T); q=np.mean(np.abs(X[0])**2)/(10**(snr/10)); ok=0
        for k in s:
            y=X[k]+(r.normal(0,np.sqrt(q/2),L)+1j*r.normal(0,np.sqrt(q/2),L))
            ok+=det(y)==k
        out[snr]=ok/T*100
    return out

def d_fold4(y):  # repo
    p=np.abs(np.fft.fft(y*DC))**2; return int(p.reshape(osf,ns).sum(0).argmax())
def d_first(y):  # pre-fix
    p=np.abs(np.fft.fft(y*DC))**2; return int(p[:ns].argmax())
def d_mf(y):     # true matched filter
    return int(np.abs(P.conj()@y).argmax())
# NOTE: the signal-bearing blocks are 0 and 3, NOT 0 and 1.  A clean symbol m puts its
# energy in bin m (block 0) and bin m-512 mod 2048 = m+1536 (block 3); blocks 1 and 2 are
# noise only -- see folding.py [1].  Earlier revisions of this file summed blocks 0 and 1,
# which added a pure-noise block and therefore UNDERSTATED what folding can do.
SIG = (0, 3)
def d_fold2(y):  # noncoherent sum of ONLY the two signal-bearing blocks
    p=np.abs(np.fft.fft(y*DC))**2; b=p.reshape(osf,ns)
    return int((b[SIG[0]]+b[SIG[1]]).argmax())
def d_coh2(y):   # coherent sum of the two signal-bearing blocks (equal gain)
    F=np.fft.fft(y*DC).reshape(osf,ns)
    return int((np.abs(F[SIG[0]]+F[SIG[1]])**2).argmax())

print("NOTE: folding 질문의 완전한 답은 folding.py 에 있다. 여기 표는 그 일부다.")
print()
for name,d in [("pre-fix power[:512]",d_first),("repo OSF-fold (4 blocks)",d_fold4),
               ("fold 2 signal blocks (noncoh)",d_fold2),("coherent sum of 2 blocks",d_coh2),
               ("matched filter (ML)",d_mf)]:
    r=run(d); print(f"{name:<32}"+"".join(f"{r[s]:>8.1f}%" for s in [-30,-25,-20,-15]))
print(f"{'':<32}"+"".join(f"{s:>8}dB" for s in [-30,-25,-20,-15]))
