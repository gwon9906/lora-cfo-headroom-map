import numpy as np
from scipy.signal import chirp, resample_poly
import numpy.matlib
exec(open('som_test.py').read().split('GS = 64')[0])   # reuse gen/awgn/spec/batch_som/purity/contiguity

def dechirp_feat(iq, snr_free=False):
    L=len(iq); ns=2**sf; osf=L//ns
    t=np.arange(L)/fs; Ts=ns/bw
    dc=np.conj(np.exp(1j*2*np.pi*(-bw/2*t+(bw/(2*Ts))*t**2)))
    d=resample_poly(iq*dc,1,osf)
    p=np.abs(np.fft.fft(d,ns))**2
    return (p/ (p.sum()+1e-12)).astype(np.float32)

def iq_feat(iq):
    v=np.concatenate([iq.real, iq.imag]).astype(np.float32)
    return v/ (np.linalg.norm(v)+1e-12)

GS=64; NG=N//GS
def run(featfn, snr, reps, tag):
    F,lab=[],[]
    for m in range(N):
        for _ in range(reps):
            F.append(featfn(X[m] if snr is None else awgn_iq(X[m],snr))); lab.append(m//GS)
    F=np.array(F); lab=np.array(lab)
    W,bmu,coord,shape=batch_som(F)
    obs,nm,ns_,z=purity_and_null(bmu,lab,shape[0]*shape[1],NG)
    cont,occ=contiguity(bmu,lab,coord,shape,NG)
    print(f"{tag:<44} purity={obs:.3f}  null={nm:.3f}  z={z:+7.1f}  contig={cont:.3f}")

print(f"{'representation / SNR':<44} {'':<10}(8 groups, chance contig=0.125)")
run(iq_feat,   None, 4, "RAW IQ      | clean (SNR=+inf)")
run(iq_feat,   -30,  4, "RAW IQ      | -30 dB")
run(dechirp_feat, None, 4, "DECHIRP+FFT | clean (SNR=+inf)")
run(dechirp_feat, -20,  4, "DECHIRP+FFT | -20 dB")
run(dechirp_feat, -25,  4, "DECHIRP+FFT | -25 dB")
run(dechirp_feat, -30,  4, "DECHIRP+FFT | -30 dB")
