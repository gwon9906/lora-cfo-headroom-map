import numpy as np
from scipy.signal import stft, chirp
import numpy.matlib
rng = np.random.default_rng(0)
sf, bw, OSF = 9, 250_000, 4
fs = bw*OSF; N = 2**sf

def gen_symbol_fs(code_word, sf, bw, Fs):
    org_Fs = Fs
    if Fs < bw: Fs = bw
    t = np.arange(0, 2**sf/bw, 1/Fs); num_samp = Fs*2**sf/bw
    ci = chirp(t, -bw/2, 2**sf/bw, bw/2, 'linear', 0)
    cq = chirp(t, -bw/2, 2**sf/bw, bw/2, 'linear', -90)
    base = numpy.matlib.repmat(ci+1j*cq, 1, 2)
    off = round((2**sf-code_word)/2**sf*num_samp)
    s = base[:, int(num_samp-off):int(num_samp-off+int(num_samp))]
    if org_Fs != Fs: s = s[:, ::int(Fs/org_Fs)]
    return s[0]

X = np.array([gen_symbol_fs(m, sf, bw, fs) for m in range(N)])

def awgn_iq(sig, snr):
    p = np.mean(np.abs(sig)**2); npw = p/(10**(snr/10))
    return sig + (rng.normal(0,np.sqrt(npw/2),sig.shape)+1j*rng.normal(0,np.sqrt(npw/2),sig.shape))

def spec_feat(iq):
    f,t,Z = stft(iq, fs=fs, window='hann', nperseg=128, noverlap=64, nfft=512,
                 boundary=None, padded=False, return_onesided=False)
    fsh = np.fft.fftshift(f); Z = np.fft.fftshift(Z, axes=0)
    Z = Z[(fsh>=-bw/2)&(fsh<bw/2), :]
    r,i = Z.real, Z.imag
    r = (r-r.min())/(r.max()-r.min()+1e-12); i = (i-i.min())/(i.max()-i.min()+1e-12)
    return np.concatenate([r.flatten(), i.flatten()])

def build(snr, reps):
    F, lab = [], []
    for m in range(N):
        for _ in range(reps):
            F.append(spec_feat(X[m] if snr is None else awgn_iq(X[m], snr)))
            lab.append(m)
    return np.array(F, dtype=np.float32), np.array(lab)

# ---------- batch SOM ----------
def batch_som(data, gx=24, gy=24, epochs=20, seed=0):
    r = np.random.default_rng(seed)
    n,d = data.shape
    W = data[r.choice(n, gx*gy)].copy()
    ii,jj = np.meshgrid(np.arange(gx), np.arange(gy), indexing='ij')
    coord = np.stack([ii.ravel(), jj.ravel()],1).astype(float)
    G2 = ((coord[:,None,:]-coord[None,:,:])**2).sum(-1)
    s0, s1 = max(gx,gy)/2.0, 0.7
    dn = (data**2).sum(1)[:,None]
    for e in range(epochs):
        sig = s0*(s1/s0)**(e/max(1,epochs-1))
        D = dn - 2*data@W.T + (W**2).sum(1)[None,:]
        bmu = np.argmin(D,1)
        H = np.exp(-G2/(2*sig**2))                    # (nodes,nodes)
        Hs = H[bmu]                                   # (n,nodes)
        num = Hs.T@data; den = Hs.sum(0)[:,None]
        W = np.where(den>1e-9, num/np.maximum(den,1e-9), W)
    D = dn - 2*data@W.T + (W**2).sum(1)[None,:]
    return W, np.argmin(D,1), coord, (gx,gy)

def purity_and_null(bmu, labels, nnodes, ngroups, nperm=200, seed=0):
    r = np.random.default_rng(seed)
    def pur(lb):
        tot = 0
        for k in range(nnodes):
            idx = np.where(bmu==k)[0]
            if len(idx)==0: continue
            tot += np.bincount(lb[idx], minlength=ngroups).max()
        return tot/len(lb)
    obs = pur(labels)
    null = np.array([pur(r.permutation(labels)) for _ in range(nperm)])
    z = (obs-null.mean())/(null.std()+1e-12)
    return obs, null.mean(), null.std(), z

def contiguity(bmu, labels, coord, shape, ngroups):
    gx,gy = shape; nn = gx*gy
    maj = -np.ones(nn, int)
    for k in range(nn):
        idx = np.where(bmu==k)[0]
        if len(idx): maj[k] = np.bincount(labels[idx], minlength=ngroups).argmax()
    same=tot=0
    grid = maj.reshape(gx,gy)
    for a in range(gx):
        for b in range(gy):
            if grid[a,b]<0: continue
            for da,db in ((0,1),(1,0)):
                a2,b2 = a+da, b+db
                if a2<gx and b2<gy and grid[a2,b2]>=0:
                    tot+=1; same += grid[a,b]==grid[a2,b2]
    return same/max(tot,1), (maj>=0).sum()/nn

GS = 64; NG = N//GS
for snr, reps, tag in [(None,4,"CLEAN (no noise, SNR=+inf)"), (-5,4,"SNR = -5 dB"), (-30,4,"SNR = -30 dB")]:
    F, sym = build(snr, reps)
    grp = sym//GS
    W,bmu,coord,shape = batch_som(F)
    obs,nm,ns,z = purity_and_null(bmu, grp, shape[0]*shape[1], NG)
    cont, occ = contiguity(bmu, grp, coord, shape, NG)
    print(f"\n=== STFT-SOM, group_size={GS} ({NG} groups) | {tag} ===")
    print(f"  samples={len(F)}  grid={shape}  occupied nodes={occ*100:.0f}%")
    print(f"  label purity        = {obs:.4f}")
    print(f"  purity under label-shuffle null = {nm:.4f} +/- {ns:.4f}   -> z = {z:+.1f}")
    print(f"  neighbour agreement (contiguity) = {cont:.4f}   (chance ~ {1/NG:.3f})")
