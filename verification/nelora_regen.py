# -*- coding: utf-8 -*-
"""
초판(cumsum 생성기)으로 낸 수치들을 해석적 생성기로 재생성하고,
검토에서 지적된 열린 항목 두 개를 닫는다.

  R1  delay-Doppler 축퇴 — 능선 기울기와 폭 (초판 §6 수치 재생성)
  R2  통째로 밀린 패킷 표 (초판 §6 수치 재생성)
  R3  브릭월 대역외 손실을 SF 별로 예측하고 관측 (표준RX - D2) 와 대조
        |상관| -> 전력 -> dB.  맞으면 '검증 안 함' 항목이 닫힌다.
  R4  D1 이 정렬에 둔감한가 (비교가 D1 에 불리하지 않다는 증거)
        정렬 전/후로 D1 이 거의 안 움직이면, 정렬을 D2 에만 유리하게 준 것이 아니다.

사용법:
  python nelora_regen.py <데이터셋루트> [--sfs 7,8,9,10]
"""
import os, glob, argparse
from collections import Counter
import numpy as np

from nelora_mf_test import norm_rows, d_mf, empirical_prototypes, parse_raw_name, load_raw, cross
from nelora_chirp import chirp_bank, base_chirp
from nelora_sync import frac_shift
from nelora_cfo import apply_corr
from nelora_stdrx import rx_standard, rx_alias, align_from_labels

SNRS = [-8, -10, -12, -14, -16, -18, -20, -22, -24]
TAUS = np.arange(-8, 8.01, 0.25)
EPSG = np.arange(-1, 1.001, 1/32)
PREAMBLE = 8


def load_sf(root, sf, proto_frac=0.33, n_test=None, seed=0):
    X, osf, M, N = chirp_bank(sf, 8)
    files = sorted(glob.glob(os.path.join(root, str(sf), '**', '*.mat'), recursive=True))
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
    if n_test and len(Traw) > n_test:
        s = np.random.default_rng(seed).choice(len(Traw), n_test, replace=False)
        Traw, tlab, tpkt = Traw[s], tlab[s], tpkt[s]
    return dict(X=X, osf=osf, M=M, N=N, bypkt=bypkt, pkts=pkts,
                Praw=Praw, plab=plab, Traw=Traw, tlab=tlab, tpkt=tpkt)


def main(root, sfs, n_test, max_clean):
    # ---------------------------------------------------------------- R3
    print('='*78)
    print('R3  브릭월 대역외 손실 예측 vs 관측 (표준RX - D2)')
    print(f"  {'SF':>3}{'대역내 |상관|':>14}{'대역내 전력':>13}{'예측 손실':>12}{'관측(초판)':>12}")
    obs = {7: 0.13, 8: 0.16, 9: 0.06, 10: 0.07}       # nelora_stdrx.py sweep 결과
    for sf in sfs:
        X, osf, M, N = chirp_bank(sf, 8)
        Z = np.fft.fft(X[0]); P = np.abs(Z)**2
        inb = (P[:N//2].sum() + P[-(N - N//2):].sum())/P.sum()
        bN = base_chirp(sf, 1)
        Zd = np.concatenate([Z[:N//2], Z[-(N - N//2):]])
        yd = np.fft.ifft(Zd)*(M/N)
        corr = abs(np.vdot(yd, bN))/np.linalg.norm(yd)/np.linalg.norm(bN)
        print(f'  {sf:>3}{corr:>14.6f}{inb:>13.6f}{-10*np.log10(inb):>+11.3f} dB'
              f'{obs.get(sf, float("nan")):>+11.2f} dB')
    print('  -> 예측이 관측 범위 안이면, 표준RX 의 0.1 dB 잔차는 물리적으로 설명된다.')

    for sf in sfs:
        d = load_sf(root, sf, n_test=n_test)
        X, osf, M, N = d['X'], d['osf'], d['M'], d['N']
        bypkt, pkts = d['bypkt'], d['pkts']
        Praw, plab = d['Praw'], d['plab']
        Traw, tlab, tpkt = d['Traw'], d['tlab'], d['tpkt']
        g_tau, g_eps = align_from_labels(Praw, plab, X)
        print('\n' + '='*78)
        print(f'SF{sf}   정렬 tau={g_tau:+.2f} 샘플, eps={g_eps:+.4f} bin   '
              f'평가 {len(np.unique(tpkt))}패킷/{len(Traw)}심볼')

        # ------------------------------------------------------------ R1
        q0 = pkts[0]
        pre = sorted([r for r in bypkt[q0] if r['symbol'] < PREAMBLE], key=lambda r: r['symbol'])
        Yp = np.array([load_raw(r['path']) for r in pre])
        Yp = Yp[[y.size == M for y in Yp]] if Yp.ndim == 1 else Yp
        K = 16
        kmax = int(round(2.0*K))
        ks = np.concatenate([np.arange(kmax+1), np.arange(-kmax, 0)])/K
        surf = []
        for tau in np.arange(-8, 8.01, 0.5):
            ps = frac_shift(X[0], tau); ps = ps/np.linalg.norm(ps)
            G = np.abs(np.fft.fft(Yp*np.conj(ps)[None, :], M*K, axis=1))
            surf.append(np.concatenate([G[:, :kmax+1], G[:, -kmax:]], axis=1).sum(0))
        surf = np.array(surf); surf /= surf.max()
        tg = np.arange(-8, 8.01, 0.5)
        near = surf > 0.99
        ti, ej = np.where(near)
        print(f'  [R1 축퇴] 최대 99% 이상인 (tau,eps) 점 {near.sum()}/{surf.size},  '
              f'tau {tg[ti].min():+.1f}~{tg[ti].max():+.1f}, eps {ks[ej].min():+.3f}~{ks[ej].max():+.3f}')
        if tg[ti].max() > tg[ti].min():
            slope = (ks[ej].max() - ks[ej].min())/(tg[ti].max() - tg[ti].min())
            print(f'            능선 기울기 {slope:+.4f} bin/샘플   (이론 1/OSF = {1/osf:+.4f})')

        # ------------------------------------------------------------ R2
        Tal = apply_corr(Traw, g_tau, g_eps)
        Xal = norm_rows(X)
        p2 = d_mf(Tal, Xal)
        bad = p2 != tlab
        tot, nb = Counter(), Counter()
        for q, b in zip(tpkt, bad):
            tot[q] += 1; nb[q] += bool(b)
        shifted = sorted([q for q in tot if nb[q]/tot[q] >= 0.5], key=int)
        print(f'  [R2 밀린 패킷] 오류율 50% 이상 {len(shifted)}/{len(tot)}: {shifted}')
        if shifted:
            print(f"    {'패킷':>6}{'n':>5}" + ''.join(f'{v:>+9d}bin' for v in range(-1, 3)))
            for q in shifted[:5]:
                m = tpkt == q
                cells = []
                for dshift in range(-1, 3):
                    Pd = norm_rows(frac_shift(X, dshift*osf))
                    cells.append((d_mf(Tal[m], Pd) != tlab[m]).mean()*100)
                print(f'  {q:>8}{m.sum():>5}' + ''.join(f'{v:11.1f}%' for v in cells))

        # ------------------------------------------------------------ R4
        by = {c: [] for c in range(N)}
        Pal = apply_corr(Praw, g_tau, g_eps)
        for y, c in zip(Pal, plab):
            if len(by[c]) < max_clean:
                by[c].append(y)
        Pemp = empirical_prototypes(by, N, M)[0]
        base_n = base_chirp(sf, 1); down = np.conj(X[0])
        grid = [v - 3*(sf - 7) for v in SNRS]

        def cr(fn, Y):
            rng = np.random.default_rng(1)
            ys = []
            for s in grid:
                sig = np.sqrt((np.abs(Y)**2).mean(1, keepdims=True)/(10.0**(s/10.0)))
                n = (rng.standard_normal(Y.shape) + 1j*rng.standard_normal(Y.shape))/np.sqrt(2)
                ys.append((fn(Y + sig*n) != tlab).mean()*100)
            return cross(grid, ys)

        print('  [R4 정렬 민감도] 정렬 전(C0) -> 정렬 후(C1), 10% SER')
        for nm, fn in (('D1 abs-alias', lambda Y: rx_alias(Y, down, osf, N, 2, 'abs')),
                       ('표준 RX', lambda Y: rx_standard(Y, base_n, osf, N, True)),
                       ('D2 이상적 MF', lambda Y: d_mf(Y, Xal))):
            a, b = cr(fn, Traw), cr(fn, Tal)
            if a is not None and b is not None:
                print(f'    {nm:<14}{a:7.2f} -> {b:7.2f} dB   변화 {b-a:+.2f} dB')
        print('    -> D1 의 변화가 작으면, 정렬을 준 것이 D1 에 불리하게 작용하지 않았다는 증거다.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--sfs', type=str, default='7,8,9,10')
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--max-clean', type=int, default=30)
    a = ap.parse_args()
    main(a.root, [int(x) for x in a.sfs.split(',')], a.n, a.max_clean)
