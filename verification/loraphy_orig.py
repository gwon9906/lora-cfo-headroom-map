# -*- coding: utf-8 -*-
"""원본 LoRaPHY (Xia et al., ACM TOSN 2022, doi:10.1145/3546869) 의 복조 경로.

NELoRa-Bench README 의 "Differences from the original code" 6번이 이렇게 쓴다:

    6. Add comparison with baseline methods, **using LoRaPhy from**
       [From Demodulation to Decoding: Toward Complete LoRa PHY Understanding
        and Implementation](https://doi.org/10.1145/3546869)

즉 그들 `decode_loraphy` 는 이 논문 구현의 포팅이라고 스스로 밝히고 있다. 그런데
원본 MATLAB 은 de-chirp 앞에 두 줄이 더 있다:

    if ~self.fast_mode
        sig = lowpass(sig, self.bw/2, self.fs);      % (1) 대역 제한 +-BW/2
    end
    self.sig = resample(sig, 2*self.bw, self.fs);    % (2) 2*BW 로 리샘플
    ...
    self.sample_num = 2*2^self.sf;                   %     심볼당 2N 샘플
    self.zero_padding_ratio = 10;                    % (3) 제로패딩 10배
    ft  = fft(sig(x:x+sample_num-1).*c, self.fft_len);
    ft_ = abs(ft(1:bin_num)) + abs(ft(end-bin_num+1:end));   % (4) 접기

`fast_mode` 는 생성자 기본값이 false 이므로 (1) 이 기본으로 걸린다.

포팅본에는 (1)(2) 가 없고 (3) 이 10 -> 100 이다. 접기 (4) 만 그대로다.

이 모듈은 (1)~(4) 를 그대로 옮긴다. 검출기 팔을 새로 만들 때의 이 프로젝트 규칙
(NELORA_REVIEW.md §7 "이상 신호 온전성 검사를 먼저 통과시킨다") 을 따라 selftest() 를 둔다.

사용법:
  python loraphy_orig.py            # 이상 신호 온전성 검사
"""
import numpy as np
from scipy.signal import firwin, lfilter

from nelora_chirp import base_chirp, chirp_bank


# -------------------------------------------------- (1) 대역 제한
def _brick(Y, M, keep_bins):
    """|f| < keep_bins*(fs/M) 만 남기는 브릭월. keep_bins 는 한쪽 폭(빈 수)."""
    Z = np.fft.fft(Y, axis=1)
    mask = np.zeros(M, dtype=bool)
    mask[:keep_bins] = True
    mask[-keep_bins:] = True
    return np.fft.ifft(Z*mask, axis=1)


def lpf_brick(Y, osf_in, N):
    """+-BW/2 브릭월 (원본의 lowpass 줄). nelora_dnn_eval.brickwall_lpf_np 와 동일.

    빈 간격은 fs/M = BW/N 이므로 한쪽 N/2 빈 = BW/2."""
    return _brick(Y, osf_in*N, N//2)


def aa_resample(Y, osf_in, N):
    """MATLAB resample() 의 반앨리어싱만. 새 나이퀴스트 = +-BW 까지 남긴다.

    lowpass 를 건너뛰어도(fast_mode=true) resample 은 이 필터를 건다. 이것을
    빠뜨리면 1 MHz 잡음이 250 kHz 로 그대로 접혀 앨리어싱 때문에 훨씬 나빠지는데,
    그것은 원본의 동작이 아니다."""
    return _brick(Y, osf_in*N, N)


def lpf_fir(Y, osf_in, N, numtaps=129):
    """MATLAB lowpass() 에 가까운 실제 FIR. 통과대역 끝 BW/2, fs = osf_in*BW.

    정규화 차단주파수는 (BW/2)/(fs/2) = (BW/2)/(osf_in*BW/2) = 1/osf_in 이다.
    복소 신호에 실계수 저역통과를 걸면 양측 |f| < BW/2 가 남는다 (MATLAB 과 동일).
    군지연 (numtaps-1)/2 를 되돌려 타이밍을 보존한다."""
    h = firwin(numtaps, cutoff=1.0/osf_in)            # = (BW/2)/(fs/2)
    d = (numtaps - 1)//2
    pad = np.concatenate([Y, np.zeros((len(Y), d), Y.dtype)], axis=1)
    return lfilter(h, 1.0, pad, axis=1)[:, d:]


# -------------------------------------------------- (1)~(4) 전체 경로
def decode_loraphy_orig(Y, sf, osf_in=8, zpr=10, lp='brick', numtaps=129):
    """원본 LoRaPHY 경로로 심볼을 복조한다. Y: (n, osf_in*N) 복소 캡처.

    (1) +-BW/2 저역통과  (2) 2*BW 로 리샘플  (3) de-chirp + zpr 배 제로패딩 FFT
    (4) |ft[:bin]| + |ft[-bin:]| 접기 -> argmax/zpr
    """
    N = 2**sf
    if lp == 'brick':
        Z = lpf_brick(Y, osf_in, N)
    elif lp == 'fir':
        Z = lpf_fir(Y, osf_in, N, numtaps)
    elif lp == 'fast':
        # fast_mode=true: lowpass 줄만 건너뛴다. resample 의 반앨리어싱(+-BW)은 남는다.
        Z = aa_resample(Y, osf_in, N)
    elif lp == 'raw':
        Z = Y                                          # 진단용: 필터 전혀 없음(앨리어싱)
    else:
        raise ValueError(lp)

    # (2) fs = osf_in*BW -> 2*BW. 'brick'/'fir' 은 이미 +-BW/2 안이고 'fast' 는
    #     +-BW 안이라, 새 나이퀴스트(+-BW) 기준으로 추가 앨리어싱이 없다.
    dec = osf_in//2
    Z = Z[:, ::dec]                                    # (n, 2N)

    # (3) de-chirp + 제로패딩 FFT
    down = np.conj(base_chirp(sf, 2))                  # 2N 샘플 다운چ프
    fft_len = 2*N*zpr
    F = np.fft.fft(Z*down[None, :], fft_len, axis=1)

    # (4) 접기: +-BW 를 반으로 접는다
    b = N*zpr
    ft = np.abs(F[:, :b]) + np.abs(F[:, -b:])
    return np.round(np.argmax(ft, axis=1)/zpr).astype(int) % N


# -------------------------------------------------- 온전성 검사
def selftest(sf=7, osf=8, ntrial=2000, seed=0):
    """타이밍/CFO = 0, 무잡음. 100% 가 아니면 구현이 틀린 것이다."""
    X, _, M, N = chirp_bank(sf, osf)
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, N, ntrial)
    Y = X[lab]*np.exp(2j*np.pi*rng.random((ntrial, 1)))   # 임의 초기위상

    print(f'SF{sf}  OSF={osf}  N={N}  M={M}  trials={ntrial}')
    print('  무잡음 · 타이밍/CFO=0 에서 100% 여야 한다.\n')
    print(f"  {'경로':<44}{'정확도':>9}")
    rows = []
    for nm, kw in [
        ('원본 LoRaPHY (LPF 브릭월 + OSF2 + zpr10)', dict(lp='brick', zpr=10)),
        ('원본 LoRaPHY (LPF FIR 129탭 + OSF2 + zpr10)', dict(lp='fir', zpr=10)),
        ('fast_mode=true (resample 반앨리어싱만)', dict(lp='fast', zpr=10)),
        ('필터 전혀 없음 (진단용, 앨리어싱)', dict(lp='raw', zpr=10)),
        ('원본 구조, zpr=100', dict(lp='brick', zpr=100)),
    ]:
        acc = (decode_loraphy_orig(Y, sf, osf, **kw) == lab).mean()
        rows.append((nm, acc))
        print(f'  {nm:<44}{acc*100:8.2f}%')
    ok = all(a > 0.9999 for _, a in rows[:1])
    print('\n  ' + ('OK — 브릭월 판이 무잡음에서 온전하다.' if ok
                    else '** 실패: 여기서 멈출 것. 아직 내 버그다.'))
    return rows


if __name__ == '__main__':
    import sys
    selftest(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
