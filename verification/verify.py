import numpy as np, sys, os
sys.path.append(r"C:/Users/gwon9/OneDrive/\ubc14\ud0d5\u0020\ud654\uba74/LoRa-bam-reconstruction-main")
from scipy.signal import stft, resample_poly

rng = np.random.default_rng(0)
sf, bw, OSF = 9, 250_000, 4
fs = bw*OSF
N = 2**sf

# ---- replicate LoRa.gen_symbol_fs (used by generate.ipynb) ----
from scipy.signal import chirp
import numpy.matlib
def gen_symbol_fs(code_word, sf, bw, down=False, Fs=None):
    if Fs is None or Fs < 0: Fs = 1000000
    org_Fs = Fs
    if Fs < bw: Fs = bw
    t = np.arange(0, 2**sf/bw, 1/Fs)
    num_samp = Fs * 2**sf/bw
    f0, f1 = -bw/2, bw/2
    chirpI = chirp(t, f0, 2**sf/bw, f1, 'linear', 0)
    chirpQ = chirp(t, f0, 2**sf/bw, f1, 'linear', -90)
    baseline = chirpI + 1j*chirpQ
    if down: baseline = np.conj(baseline)
    baseline = numpy.matlib.repmat(baseline,1,2)
    offset = round((2**sf - code_word) / 2**sf * num_samp)
    symb = baseline[:, int(num_samp-offset):int(num_samp-offset+int(num_samp))]
    if org_Fs != Fs:
        symb = symb[:, ::int(Fs/org_Fs)]
    return symb[0]

X = np.array([gen_symbol_fs(m, sf, bw, Fs=fs) for m in range(N)])
print("clean IQ shape", X.shape)

# =========== TEST A: pairwise geometry of clean symbols in IQ space ===========
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
G = np.abs(Xn @ Xn.conj().T)          # |cosine similarity|
off = G[~np.eye(N, dtype=bool)]
print("\n=== A. Clean LoRa symbols in raw-IQ space ===")
print(f"  |cos| between DIFFERENT symbols: mean={off.mean():.5f} max={off.max():.5f} min={off.min():.5f}")
D = np.sqrt(np.maximum(0, 2 - 2*np.real(Xn @ Xn.conj().T)))
offD = D[~np.eye(N, dtype=bool)]
print(f"  normalized L2 distance:          mean={offD.mean():.5f} std={offD.std():.5f} "
      f"min={offD.min():.5f} max={offD.max():.5f}")
print(f"  -> ratio max/min = {offD.max()/offD.min():.4f}  (1.0 == perfectly equidistant)")

# is symbol index correlated with distance?
d_adj = np.mean([D[m, m+1] for m in range(N-1)])
d_far = np.mean([D[m, (m+256) % N] for m in range(N)])
print(f"  mean dist to NEIGHBOUR symbol (m,m+1)   = {d_adj:.5f}")
print(f"  mean dist to OPPOSITE symbol (m,m+256)  = {d_far:.5f}")
