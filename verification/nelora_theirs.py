# -*- coding: utf-8 -*-
"""
그들의 평가 경로를 그대로 재현하고, 축 차이를 수치로 잡는다.

daibiaoxuwu/NeLoRa_Dataset (NELoRa-Bench, ICLR'23 workshop) main.py 에서 확인한 것:

  def decode_loraphy(data_in, num_classes, downchirp):
      upsampling = 100                                  # <- 내 D1 은 U=1 로 쟀다
      chirp_data = data_in * downchirp                  # OSF=8 에서 de-chirp
      fft_raw = fft(chirp_data, len(chirp_data) * upsampling)
      target_nfft = num_classes * upsampling
      cut1 = fft_raw[:target_nfft]; cut2 = fft_raw[-target_nfft:]
      return round(argmax(abs(cut1) + abs(cut2)) / upsampling) % num_classes

  - 데시메이션 없음.  이 보고서의 전제는 유지된다.
  - upsampling=100.  내가 '버그' 라며 제외했던 U 파라미터가 그들 기본값이다. 다시 잰다.
  - downchirp 은 scipy.signal.chirp (해석적 2차 위상) = nelora_chirp 와 동일
    (상수 위상 j 차이뿐). 그들 생성기는 멀쩡했고 내 cumsum 판이 틀렸던 것.
  - test() 는 plt.axhline(y=0.9), 즉 10% SER 임계. 내 임계와 같다.
  - SNR 정의가 다르다:
        그들:  amp = 10^(-snr/20) * mean(|y|),  noise = amp/sqrt(2)*(randn + j randn)
               => 신호 대용량이 (mean|y|)^2   [평균 진폭]
        내 것:  sigma = sqrt(mean(|y|^2) / 10^(snr/10))
               => 신호량이 mean(|y|^2)        [평균 전력]
    일정 포락선 신호면 같지만, 진폭이 흔들리는 실캡처에서는 다르다. 그 차이를 측정한다.

사용법:
  python nelora_theirs.py <데이터폴더> [--sf 7]
"""
import os, glob, argparse
import numpy as np

from nelora_mf_test import norm_rows, d_mf, empirical_prototypes, parse_raw_name, load_raw, cross
from nelora_chirp import chirp_bank, base_chirp
from nelora_cfo import apply_corr
from nelora_stdrx import rx_standard, align_from_labels

CHUNK = 64          # U=100 FFT 는 메모리를 많이 먹는다


# ------------------------------------------------- 그들 코드 그대로
def decode_loraphy(Y, down, N, U=100, chunk=CHUNK):
    """NELoRa-Bench main.py 의 decode_loraphy 를 벡터화한 것. 로직은 동일."""
    M = Y.shape[1]
    out = np.empty(len(Y), dtype=int)
    T = N*U
    for i in range(0, len(Y), chunk):
        Z = np.fft.fft(Y[i:i+chunk]*down[None, :], M*U, axis=1)
        s = np.abs(Z[:, :T]) + np.abs(Z[:, -T:])
        out[i:i+chunk] = np.round(np.argmax(s, 1)/U).astype(int) % N
    return out


def add_noise_theirs(Y, snr_db, rng):
    """그들 add_noise: 평균 '진폭' 기준."""
    amp = (10.0**(-snr_db/20.0))*np.abs(Y).mean(1, keepdims=True)
    n = rng.standard_normal(Y.shape) + 1j*rng.standard_normal(Y.shape)
    return Y + (amp/np.sqrt(2))*n


def add_noise_mine(Y, snr_db, rng):
    """내 add_awgn: 평균 '전력' 기준."""
    sig = np.sqrt((np.abs(Y)**2).mean(1, keepdims=True)/(10.0**(snr_db/10.0)))
    n = (rng.standard_normal(Y.shape) + 1j*rng.standard_normal(Y.shape))/np.sqrt(2)
    return Y + sig*n


def axis_offset_db(Y):
    """두 SNR 정의의 차이(dB): 10log10( mean|y|^2 / (mean|y|)^2 ). 0 이면 축이 같다."""
    p = (np.abs(Y)**2).mean(1)
    a = np.abs(Y).mean(1)**2
    return 10*np.log10(p/a)


def sweep(fn, Y, lab, snrs, noise, seed=1):
    rng = np.random.default_rng(seed)
    return {s: (fn(noise(Y, s, rng)) != lab) for s in snrs}


def crossings(err, snrs, thresholds=(10.0, 20.0, 30.0), sel=None):
    ys = [err[s][sel].mean()*100 if sel is not None else err[s].mean()*100 for s in snrs]
    return {t: cross(list(snrs), ys, t) for t in thresholds}


def main(root, sf, n_test, proto_frac, max_clean, nboot, upsampling,
         boot_u=4, only='ABCDE'):
    X, osf, M, N = chirp_bank(sf, 8)
    Xn = norm_rows(X); base_n = base_chirp(sf, 1); down = np.conj(X[0])

    files = sorted(glob.glob(os.path.join(root, '**', '*.mat'), recursive=True))
    recs = [m for m in (parse_raw_name(f) for f in files) if m and m['sf'] == sf]
    bypkt = {}
    for r in recs:
        bypkt.setdefault(r['packet'], []).append(r)
    pkts = sorted(bypkt, key=int)
    npro = max(1, int(round(len(pkts)*proto_frac)))
    pro = set(pkts[:npro])

    def payload(pset):
        A, L, Q = [], [], []
        for q in pset:
            for r in [x for x in bypkt[q] if x['symbol'] >= 12]:
                y = load_raw(r['path'])
                if y.size == M:
                    A.append(y); L.append(r['code_label'] % N); Q.append(q)
        return np.array(A), np.array(L), np.array(Q)

    Praw, plab, _ = payload([q for q in pkts if q in pro])
    Traw, tlab, tpkt = payload([q for q in pkts if q not in pro])
    rng = np.random.default_rng(0)
    if len(Traw) > n_test:
        s = rng.choice(len(Traw), n_test, replace=False)
        Traw, tlab, tpkt = Traw[s], tlab[s], tpkt[s]
    g_tau, g_eps = align_from_labels(Praw, plab, X)
    Tal = apply_corr(Traw, g_tau, g_eps)
    Pal = apply_corr(Praw, g_tau, g_eps)
    by = {c: [] for c in range(N)}
    for y, c in zip(Pal, plab):
        if len(by[c]) < max_clean:
            by[c].append(y)
    Pemp = empirical_prototypes(by, N, M)[0]
    print(f'SF{sf}  평가 {len(np.unique(tpkt))}패킷/{len(Tal)}심볼  정렬 tau={g_tau:+.2f} eps={g_eps:+.4f}')

    # ---------------------------------------------------------------- A
    print('\n' + '='*78)
    print('A  SNR 축 차이 — 그들 정의(평균 진폭) vs 내 정의(평균 전력)')
    off = axis_offset_db(Traw)
    print(f'  10log10( mean|y|^2 / (mean|y|)^2 ) = {off.mean():+.3f} dB '
          f'(중앙값 {np.median(off):+.3f}, 5~95% {np.percentile(off,5):+.3f}~{np.percentile(off,95):+.3f})')
    print(f'  -> 같은 잡음에 대해 내 SNR 표기가 그들보다 {off.mean():+.2f} dB 높다.')
    print(f'     즉 내 곡선을 그들 축으로 옮기려면 {-off.mean():+.2f} dB 평행이동해야 한다.')
    ideal = X[:64]
    print(f'  참고: 이상적(일정 포락선) chirp 에서는 {axis_offset_db(ideal).mean():+.4f} dB (0 이어야 정상)')

    # ---------------------------------------------------------------- B
    print('\n' + '='*78)
    print(f'B  upsampling 재검: 그들 기본값은 U=100 이다 (내 초판은 U=1)')
    snrs = [-8, -10, -12, -14, -16, -18, -20, -22, -24]
    for U in (1, 2, 4, 10, upsampling):
        e = sweep(lambda Y: decode_loraphy(Y, down, N, U), Tal, tlab, snrs, add_noise_mine)
        c = crossings(e, snrs)
        print(f'  U={U:4d}  10% SER {c[10.0]:7.2f} dB   '
              f'SER@-14 {e[-14].mean()*100:5.1f}%  @-18 {e[-18].mean()*100:5.1f}%')
    print('  -> U 를 올려도 거의 안 변하면, 초판이 U=1 로 잰 것은 문제가 아니었다.')

    # ---------------------------------------------------------------- C
    print('\n' + '='*78)
    print('C  같은 축·같은 임계에서 검출기 비교')
    arms = [(f'D1 decode_loraphy (U={upsampling}, 그들 코드)',
             lambda Y: decode_loraphy(Y, down, N, upsampling)),
            ('표준 RX (LPF->데시->dechirp)', lambda Y: rx_standard(Y, base_n, osf, N, True)),
            ('D2 이상적 MF', lambda Y: d_mf(Y, Xn)),
            ('D3 경험적 MF', lambda Y: d_mf(Y, Pemp))]
    for nzname, nz in (('내 정의(평균 전력)', add_noise_mine), ('그들 정의(평균 진폭)', add_noise_theirs)):
        print(f'\n  [{nzname}]   임계별 도달 SNR')
        print(f"    {'검출기':<34}{'10%':>9}{'20%':>9}{'30%':>9}")
        store = {}
        for nm, fn in arms:
            e = sweep(fn, Tal, tlab, snrs, nz)
            store[nm] = e
            c = crossings(e, snrs)
            print(f'    {nm:<34}' + ''.join(
                f'{("없음" if c[t] is None else f"{c[t]:7.2f}"):>9}' for t in (10.0, 20.0, 30.0)))
        d1k, stk = arms[0][0], arms[1][0]
        print(f'    {"-> D1 - 표준RX":<34}' + ''.join(
            f'{(crossings(store[d1k],snrs)[t] - crossings(store[stk],snrs)[t]):>+9.2f}'
            if crossings(store[d1k],snrs)[t] and crossings(store[stk],snrs)[t] else f'{"-":>9}'
            for t in (10.0, 20.0, 30.0)))

    # ---------------------------------------------------------------- D
    print('\n' + '='*78)
    print(f'D  부트스트랩 — 패킷 재표집 + 잡음 재추출 ({nboot} 복제)')
    print('   (초판은 잡음을 고정한 대응비교였다. 여기서는 복제마다 잡음을 새로 뽑는다.)')
    upk, inv = np.unique(tpkt, return_inverse=True)
    idx_by = [np.where(inv == i)[0] for i in range(len(upk))]
    rgb = np.random.default_rng(11)
    gaps = {'D1-표준RX': [], '표준RX-D2': [], 'D2-D3': []}
    for b in range(nboot):
        sel = np.concatenate([idx_by[i] for i in rgb.integers(0, len(upk), len(upk))])
        Yb, lb = Tal[sel], tlab[sel]
        rr = np.random.default_rng(10_000 + b)          # 복제마다 새 잡음
        ys = {}
        for nm, fn in (('D1', lambda Y: decode_loraphy(Y, down, N, boot_u)),
                       ('STD', lambda Y: rx_standard(Y, base_n, osf, N, True)),
                       ('D2', lambda Y: d_mf(Y, Xn)),
                       ('D3', lambda Y: d_mf(Y, Pemp))):
            r2 = np.random.default_rng(10_000 + b)      # 네 팔이 같은 잡음을 본다
            ys[nm] = cross(snrs, [(fn(add_noise_mine(Yb, s, r2)) != lb).mean()*100 for s in snrs])
        if ys['D1'] and ys['STD']:
            gaps['D1-표준RX'].append(ys['D1'] - ys['STD'])
        if ys['STD'] and ys['D2']:
            gaps['표준RX-D2'].append(ys['STD'] - ys['D2'])
        if ys['D2'] and ys['D3']:
            gaps['D2-D3'].append(ys['D2'] - ys['D3'])
    for k, v in gaps.items():
        if v:
            a = np.array(v)
            lo, hi = np.percentile(a, 2.5), np.percentile(a, 97.5)
            sig = '유의' if (lo > 0 or hi < 0) else '0 과 구별 안 됨'
            print(f'  {k:<12}{np.median(a):+7.2f} dB  [{lo:+6.2f}, {hi:+6.2f}]  {sig}  (n={len(a)})')

    # ---------------------------------------------------------------- E
    print('\n' + '='*78)
    print('E  표준RX - D2 잔차가 브릭월 대역외 손실로 설명되는가')
    Z = np.fft.fft(X[0]); P = np.abs(Z)**2
    inb = (P[:N//2].sum() + P[-(N - N//2):].sum())/P.sum()
    print(f'  SF{sf}: 대역내 전력비 {inb:.6f}  ->  예상 손실 {-10*np.log10(inb):+.3f} dB')
    print('  (관측된 표준RX - D2 와 맞으면 §7-3 은 닫힌다)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--sf', type=int, default=7)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--proto-frac', type=float, default=0.33)
    ap.add_argument('--max-clean', type=int, default=30)
    ap.add_argument('--boot', type=int, default=200)
    ap.add_argument('--upsampling', type=int, default=100)
    ap.add_argument('--boot-u', type=int, default=4, help='부트스트랩용 U (비용)')
    ap.add_argument('--only', type=str, default='ABCDE')
    a = ap.parse_args()
    main(a.root, a.sf, a.n, a.proto_frac, a.max_clean, a.boot, a.upsampling,
         a.boot_u, a.only)
