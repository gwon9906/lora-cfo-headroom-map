# -*- coding: utf-8 -*-
"""
올바른 LoRa chirp 생성기.

nelora_mf_test.ideal_chirps 는 순시주파수를 cumsum 으로 적분해 위상을 만든다.
그 이산화 오차 때문에

  - OSF=8 판을 데시메이션한 것과 OSF=1 판이 |상관| 0.72 밖에 안 되고
    (그래서 '표준 수신기' 를 짜면 무잡음에서도 23% 밖에 못 맞춘다)
  - X[c] 가 roll(X[0], -OSF*c) 와 일치하지 않는다 (최대오차 2.0)

LoRa chirp 의 위상은 닫힌 형태로 알려져 있으므로 그대로 쓴다.
기저 업چ프는 t in [0, T), T = N/BW 에서 f(t) = -BW/2 + (BW/T) t 이므로

  phase(t) = 2*pi*( -BW/2 * t + BW/(2T) * t^2 )

fs = OSF*BW, t = n/fs 를 넣으면 샘플율에 무관한 형태가 된다:

  phase(n) = 2*pi*( n^2/(2*N*OSF^2) - n/(2*OSF) )

코드 c 는 기저의 순환시프트다:  X_c[n] = base[(n + c*OSF) mod M].

이 규약이면 자동으로:
  base_M[OSF*k] == base_N[k]                       (데시메이션이 정확)
  X_c 를 데시메이션하면 base_N 의 c 만큼 순환시프트   (표준 수신기가 성립)
"""
import numpy as np


def base_chirp(sf, osf):
    """기저 업چ프 한 개. 해석적 2차 위상이라 어떤 OSF 에서도 서로 일관된다."""
    N = 2**sf
    M = osf*N
    n = np.arange(M)
    return np.exp(2j*np.pi*(n**2/(2.0*N*osf**2) - n/(2.0*osf)))


def chirp_bank(sf, osf):
    """코드 0..N-1 의 업چ프 뱅크. X_c = base 를 c*osf 만큼 순환시프트."""
    N = 2**sf
    M = osf*N
    base = base_chirp(sf, osf)
    idx = (np.arange(M)[None, :] + osf*np.arange(N)[:, None]) % M
    return base[idx], osf, M, N


def selftest(sf=7, osf=8):
    """이 모듈이 주장하는 성질을 실제로 만족하는지 확인한다."""
    X, osf_, M, N = chirp_bank(sf, osf)
    bN = base_chirp(sf, 1)
    out = {}

    # 1) 순환시프트 일관성
    Xr = np.array([np.roll(X[0], -osf*c) for c in range(N)])
    out['shift_err'] = float(np.abs(X - Xr).max())

    # 2) 데시메이션 정확도: base_M[osf*k] == base_N[k]
    out['decim_err'] = float(np.abs(X[0][::osf] - bN).max())

    # 3) 브릭월 데시메이션도 같은 결과를 주는가
    Z = np.fft.fft(X[0])
    Zd = np.concatenate([Z[:N//2], Z[-(N - N//2):]])
    yd = np.fft.ifft(Zd)*(M/N)
    out['brick_corr'] = float(abs(np.vdot(yd, bN))/np.linalg.norm(yd)/np.linalg.norm(bN))

    # 4) 뱅크 직교성 (서로 다른 코드끼리 얼마나 구분되는가)
    Xn = X/np.linalg.norm(X, axis=1, keepdims=True)
    G = np.abs(Xn @ np.conj(Xn).T)
    np.fill_diagonal(G, 0)
    out['max_xcorr'] = float(G.max())

    # 5) 데시메이션 후 코드 c 가 base_N 의 c 순환시프트인가
    XdZ = np.fft.fft(X, axis=1)
    Xd = np.fft.ifft(np.concatenate([XdZ[:, :N//2], XdZ[:, -(N - N//2):]], axis=1), axis=1)
    ok = []
    for c in range(0, N, max(1, N//8)):
        ref = np.roll(bN, -c)
        ok.append(abs(np.vdot(Xd[c], ref))/np.linalg.norm(Xd[c])/np.linalg.norm(ref))
    out['decim_shift_corr'] = float(np.min(ok))
    return out


if __name__ == '__main__':
    import sys
    from nelora_mf_test import ideal_chirps, norm_rows
    sf = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    print(f'SF{sf}  올바른 생성기 자체검사')
    r = selftest(sf)
    print(f"  순환시프트 최대오차        {r['shift_err']:.3e}   (0 이어야 함)")
    print(f"  데시메이션 최대오차        {r['decim_err']:.3e}   (0 이어야 함)")
    print(f"  브릭월 데시메이션 |상관|   {r['brick_corr']:.6f}   (1 이어야 함)")
    print(f"  데시메 후 시프트 |상관|    {r['decim_shift_corr']:.6f}   (1 이어야 함)")
    print(f"  코드 간 최대 |상호상관|    {r['max_xcorr']:.4f}")

    print(f'\nSF{sf}  기존 cumsum 생성기와 비교')
    Pold, osf, M, N = ideal_chirps(sf, 125e3)
    Pnew = chirp_bank(sf, osf)[0]
    c = np.abs(np.sum(norm_rows(Pold)*np.conj(norm_rows(Pnew)), 1))
    print(f'  두 생성기 간 |상관|: 평균 {c.mean():.4f}, 최소 {c.min():.4f}')
    Xr = np.array([np.roll(Pold[0], -osf*k) for k in range(N)])
    print(f'  cumsum 판의 순환시프트 최대오차 {np.abs(Pold - Xr).max():.3f}')
    bN_old = ideal_chirps(sf, 125e3, fs=125e3)[0][0]
    v = abs(np.vdot(Pold[0][::osf], bN_old))/np.linalg.norm(Pold[0][::osf])/np.linalg.norm(bN_old)
    print(f'  cumsum 판의 데시메이션 |상관| {v:.4f}   <- 표준 수신기가 깨진 원인')
