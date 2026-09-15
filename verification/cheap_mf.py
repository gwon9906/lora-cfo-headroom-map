"""
Q: 최적 검출기(MF)는 엣지 기기에서 못 돌릴 만큼 비싼가?
A: 아니다. X[m] = roll(X[0], -osf*m) 이므로 MF 뱅크 전체는 순환상관이고,
   FFT 2번 + 곱 1번으로 '정확히' 계산된다. 512x2048 직접 상관은 불필요.
   => 정확도/연산량 트레이드오프는 존재하지 않는다. 5 dB 는 공짜로 회수된다.
"""
import io, numpy as np, time
exec(io.open('cfo_headroom.py', encoding='utf-8').read().split('def rx(')[0])
L = X.shape[1]; osf = L//N
tt = np.arange(L)/fs; Ts = N/bw
DC = np.conj(np.exp(1j*2*np.pi*(-bw/2*tt+(bw/(2*Ts))*tt**2)))

assert all(np.allclose(X[m], np.roll(X[0], -osf*m), atol=1e-5) for m in [1,7,100,511])
F0c = np.conj(np.fft.fft(X[0]))          # 한 번만 계산
IDX = (-osf*np.arange(N)) % L            # 심볼 m 에 대응하는 lag

def d_direct(y):  return int(np.abs(P.conj()@y).argmax())          # 512x2048 MAC
def d_fft(y):                                                       # FFT 2회
    return int(np.abs(np.fft.ifft(np.fft.fft(y)*F0c)[IDX]).argmax())
def d_repo(y):                                                      # FFT 1회
    return int((np.abs(np.fft.fft(y*DC))**2).reshape(osf,N).sum(0).argmax())

r = np.random.default_rng(11); T = 3000
print(f"identical noise realisations, {T} trials/SNR\n")
print(f"{'SNR':>5}{'repo dechirp+FFT':>19}{'FFT-MF (optimal)':>19}{'direct MF':>12}{'agree':>8}")
for snr in [-30,-25,-20,-15]:
    s=r.integers(0,N,T); q=np.mean(np.abs(X[0])**2)/(10**(snr/10)); a=b=c=ag=0
    for k in s:
        y=X[k]+(r.normal(0,np.sqrt(q/2),L)+1j*r.normal(0,np.sqrt(q/2),L))
        da,db,dc=d_direct(y),d_fft(y),d_repo(y)
        a+=da==k; b+=db==k; c+=dc==k; ag+=da==db
    print(f"{snr:>5}{c/T*100:>18.1f}%{b/T*100:>18.1f}%{a/T*100:>11.1f}%{ag/T*100:>7.0f}%")

y=X[3]+0j; n=3000
print("\ncost per symbol")
for name,f in [("repo dechirp+FFT",d_repo),("FFT-MF (optimal)",d_fft),("direct MF bank",d_direct)]:
    t0=time.perf_counter()
    for _ in range(n): f(y)
    print(f"  {name:<22}{(time.perf_counter()-t0)/n*1e6:9.1f} us")
