import io, numpy as np
exec(io.open('cfo_headroom.py', encoding='utf-8').read().split('SNRS = ')[0])
GRID=np.arange(-0.5,0.501,0.125)

def acc_packet(Y,syms,grid,ns):
    npk=len(Y)//ns; ok=0
    for k in range(npk):
        sl=slice(k*ns,(k+1)*ns); Yp,sp=Y[sl],syms[sl]
        bf,bs=None,-np.inf
        for f in grid:
            s=mf_scores(Yp,f).max(1).sum()
            if s>bs: bs,bf=s,f
        ok+=(mf_scores(Yp,bf).argmax(1)==sp).sum()
    return ok/(npk*ns)*100

T=1024
print("[C] does packet-pooling still reach the genie bound at the deepest SNR?")
print(f"{'SNR':>5} |{'blind':>8}{'ns=32':>8}{'ns=128':>9}{'ns=512':>9}{'genie':>8} | gap(genie-best)")
print("-"*64)
for snr in [-32,-30,-28,-26]:
    s=rng.integers(0,N,T); eps=np.repeat(np.round(rng.uniform(-0.5,0.5,T//512),3),512)
    Y=rx(s,snr,eps)
    ab=acc_blind(Y,s); ag=acc_genie(Y,s,eps)
    a32=acc_packet(Y,s,GRID,32); a128=acc_packet(Y,s,GRID,128); a512=acc_packet(Y,s,GRID,512)
    best=max(a32,a128,a512)
    print(f"{snr:>5} |{ab:>7.1f}%{a32:>7.1f}%{a128:>8.1f}%{a512:>8.1f}%{ag:>7.1f}% | {ag-best:>6.1f}%p")

# ---- CFO drift (SFO-like): CFO is NOT constant across the packet ----
print("\n[D] CFO drift within packet (constant-CFO estimator model mismatch), ns=128, SNR=-25")
print(f"{'drift over pkt (bins)':>24} |{'blind':>8}{'pkt ns=128':>12}{'genie':>8}")
print("-"*56)
for drift in [0.0, 0.1, 0.25, 0.5, 1.0]:
    s=rng.integers(0,N,T)
    base=np.repeat(np.round(rng.uniform(-0.5,0.5,T//128),3),128)
    ramp=np.tile(np.linspace(-drift/2,drift/2,128),T//128)
    eps=base+ramp
    Y=rx(s,-25,eps)
    print(f"{drift:>24.2f} |{acc_blind(Y,s):>7.1f}%{acc_packet(Y,s,GRID,128):>11.1f}%{acc_genie(Y,s,eps):>7.1f}%")
