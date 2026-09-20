import io, numpy as np

src = io.open('cfo_headroom.py', encoding='utf-8').read().split('SNRS = ')[0]
exec(src)

SNRS=[-30,-25,-20,-15]; GRID=np.arange(-0.5,0.501,0.125)

def blockwise_eps(T, ns):
    return np.repeat(np.round(rng.uniform(-0.5,0.5,T//ns),3), ns)

def acc_packet(Y, syms, grid, ns):
    T=len(Y); npk=T//ns; ok=0
    for k in range(npk):
        sl=slice(k*ns,(k+1)*ns); Yp,sp=Y[sl],syms[sl]
        bf,bs=None,-np.inf
        for f in grid:
            s=mf_scores(Yp,f).max(1).sum()
            if s>bs: bs,bf=s,f
        ok += (mf_scores(Yp,bf).argmax(1)==sp).sum()
    return ok/(npk*ns)*100

T=12800

print("[B-fix] residual CFO ~U(-0.5,0.5) bin, CONSTANT within a packet (physical CFO model)")
print("        %d trials/cell.  REAL headroom = genie - pkt128 (best classical)" % T)
hdr=(f"{'SNR':>5} |{'blind MF':>10}{'joint sym':>11}{'pkt ns=8':>10}{'pkt ns=32':>11}"
     f"{'pkt ns=128':>12}{'genie':>9} |{'vs blind':>12}{'REAL(g-p128)':>16}")
print(hdr); print("-"*len(hdr))
for snr in SNRS:
    s=rng.integers(0,N,T); eps=blockwise_eps(T,128)
    Y=rx(s,snr,eps)
    ab=acc_blind(Y,s); ag=acc_genie(Y,s,eps); aj=acc_joint(Y,s,GRID)
    p8=acc_packet(Y,s,GRID,8); p32=acc_packet(Y,s,GRID,32); p128=acc_packet(Y,s,GRID,128)
    print(f"{snr:>5} |{ab:>9.1f}%{aj:>10.1f}%{p8:>9.1f}%{p32:>10.1f}%{p128:>11.1f}%{ag:>8.1f}%"
          f" |{ag-ab:>10.1f}%p{ag-p128:>14.1f}%p")