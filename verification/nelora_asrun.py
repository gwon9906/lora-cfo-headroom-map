# -*- coding: utf-8 -*-
"""
그들 파이프라인을 '있는 그대로' 재현하고, 같은 조건에서 표준 수신기를 붙인다.

NELoRa-Bench main.py 에서 확인한 그들 평가 조건 네 가지:

  1. load_data() 가 데이터셋을 거른다:
         if decode_loraphy(chirp_raw, num_classes, downchirp) == truth_idx:
     무잡음 캡처에서 baseline 이 이미 맞히는 심볼만 남긴다.
     -> 내가 찾은 5.9% 바닥(+-1 bin 밀린 패킷)은 그들 집합에 아예 없다.
     -> baseline 은 무한 SNR 에서 구성상 100% 다.
  2. upsampling=100 (내 초판은 U=1. 정렬된 데이터에서 U=100 은 U=1 보다 1.1 dB 나쁘다)
  3. 정렬(타이밍/CFO 보정)을 하지 않는다
  4. SNR 은 평균 '진폭' 기준, 임계는 10% SER (plt.axhline(y=0.9))

여기서는 필터 적용/미적용 두 경우를 모두 재고, 임계도 10/20/30% 를 함께 본다.
10% 만 보면 바닥 근처라 교차점이 불안정하다(필터 없을 때 특히).

사용법:
  python nelora_asrun.py <데이터폴더> [--sf 7] [--n 3000]
"""
import os, glob, argparse
import numpy as np

from nelora_mf_test import norm_rows, d_mf, empirical_prototypes, parse_raw_name, load_raw, cross
from nelora_chirp import chirp_bank, base_chirp
from nelora_cfo import apply_corr
from nelora_stdrx import rx_standard, align_from_labels
from nelora_theirs import decode_loraphy, add_noise_theirs, add_noise_mine

THR = (10.0, 20.0, 30.0)


def crossings(errs, snrs, sel=None):
    ys = [(errs[s][sel] if sel is not None else errs[s]).mean()*100 for s in snrs]
    return {t: cross(list(snrs), ys, t) for t in THR}


def fmt(c):
    return ''.join(f'{("없음" if c[t] is None else f"{c[t]:7.2f}"):>9}' for t in THR)


def main(root, sf, n_test, proto_frac, max_clean, U, nboot):
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
    print(f'SF{sf}  프로토타입 {len(Praw)}심볼 / 평가 {len(Traw)}심볼 ({len(np.unique(tpkt))}패킷)')

    # --- 그들 필터: 무잡음에서 decode_loraphy 가 맞히는 심볼만
    print(f'\n그들 load_data() 필터 적용 중 (decode_loraphy U={U}, 무잡음)...')
    keepT = decode_loraphy(Traw, down, N, U) == tlab
    keepP = decode_loraphy(Praw, down, N, U) == plab
    print(f'  평가셋: {keepT.sum()}/{len(keepT)} 통과 ({keepT.mean()*100:.2f}%)  '
          f'-> {(~keepT).sum()}심볼이 그들 데이터셋에서 빠진다')
    print(f'  버려지는 심볼이 몰린 패킷: ', end='')
    lost = {}
    for q, k in zip(tpkt, keepT):
        lost[q] = lost.get(q, [0, 0])
        lost[q][0] += (not k); lost[q][1] += 1
    worst = sorted(lost.items(), key=lambda kv: -kv[1][0]/kv[1][1])[:5]
    print(', '.join(f'pkt{q}({a}/{b})' for q, (a, b) in worst))

    rng = np.random.default_rng(0)
    snrs = [-6, -8, -10, -12, -14, -16, -18, -20, -22, -24]

    def sub(Y, L, P, mask, cap):
        Y, L, P = Y[mask], L[mask], P[mask]
        if len(Y) > cap:
            s = rng.choice(len(Y), cap, replace=False)
            Y, L, P = Y[s], L[s], P[s]
        return Y, L, P

    for tag, mT, mP in (('필터 적용 (그들 조건)', keepT, keepP),
                        ('필터 없음 (전체 심볼)', np.ones(len(keepT), bool), np.ones(len(keepP), bool))):
        Yt, Lt, Qt = sub(Traw, tlab, tpkt, mT, n_test)
        Yp, Lp = Praw[mP], plab[mP]
        g_tau, g_eps = align_from_labels(Yp, Lp, X)
        Yal = apply_corr(Yt, g_tau, g_eps)
        Pal = apply_corr(Yp, g_tau, g_eps)
        by = {c: [] for c in range(N)}
        for y, c in zip(Pal, Lp):
            if len(by[c]) < max_clean:
                by[c].append(y)
        Pemp = empirical_prototypes(by, N, M)[0]

        print('\n' + '='*80)
        print(f'{tag}   평가 {len(Yt)}심볼 / {len(np.unique(Qt))}패킷   '
              f'정렬 tau={g_tau:+.2f} eps={g_eps:+.4f}')
        print(f'  무잡음 오류: D1(U={U}) {(decode_loraphy(Yt, down, N, U) != Lt).mean()*100:.2f}%   '
              f'표준RX {(rx_standard(Yal, base_n, osf, N, True) != Lt).mean()*100:.2f}%')

        arms = [
            (f'D1 decode_loraphy U={U} 정렬없음 (그들 as-run)',
             lambda Y: decode_loraphy(Y, down, N, U), False),
            (f'D1 decode_loraphy U=1 정렬없음',
             lambda Y: decode_loraphy(Y, down, N, 1), False),
            ('표준 RX 정렬없음',
             lambda Y: rx_standard(Y, base_n, osf, N, True), False),
            ('표준 RX 정렬적용',
             lambda Y: rx_standard(Y, base_n, osf, N, True), True),
            ('D2 이상적 MF 정렬적용', lambda Y: d_mf(Y, Xn), True),
            ('D3 경험적 MF 정렬적용', lambda Y: d_mf(Y, Pemp), True),
        ]
        print(f"\n  [그들 SNR 정의: 평균 진폭]   {'검출기':<38}" + ''.join(f'{t:>8.0f}%' for t in THR))
        store = {}
        for nm, fn, al in arms:
            rg = np.random.default_rng(1)
            Y0 = Yal if al else Yt
            e = {s: (fn(add_noise_theirs(Y0, s, rg)) != Lt) for s in snrs}
            store[nm] = e
            print(f'  {"":<26}{nm:<38}' + fmt(crossings(e, snrs)))

        a, b = arms[0][0], arms[3][0]
        ca, cb = crossings(store[a], snrs), crossings(store[b], snrs)
        print(f'  {"":<26}{"-> 그들 as-run D1 - 표준RX(정렬)":<38}' + ''.join(
            f'{(ca[t]-cb[t]):>+9.2f}' if (ca[t] is not None and cb[t] is not None) else f'{"-":>9}'
            for t in THR))
        c0 = crossings(store[arms[2][0]], snrs)
        print(f'  {"":<26}{"-> 그들 as-run D1 - 표준RX(정렬없음)":<38}' + ''.join(
            f'{(ca[t]-c0[t]):>+9.2f}' if (ca[t] is not None and c0[t] is not None) else f'{"-":>9}'
            for t in THR))

        # 부트스트랩: 패킷 재표집 + 복제마다 새 잡음
        if nboot:
            upk, inv = np.unique(Qt, return_inverse=True)
            idx_by = [np.where(inv == i)[0] for i in range(len(upk))]
            rgb = np.random.default_rng(11)
            acc = {t: [] for t in THR}
            for bi in range(nboot):
                sel = np.concatenate([idx_by[i] for i in rgb.integers(0, len(upk), len(upk))])
                r1 = np.random.default_rng(50_000 + bi)
                ya = [(decode_loraphy(add_noise_theirs(Yt[sel], s, r1), down, N, U) != Lt[sel]).mean()*100
                      for s in snrs]
                r2 = np.random.default_rng(50_000 + bi)
                yb = [(rx_standard(add_noise_theirs(Yal[sel], s, r2), base_n, osf, N, True) != Lt[sel]).mean()*100
                      for s in snrs]
                for t in THR:
                    va, vb = cross(snrs, ya, t), cross(snrs, yb, t)
                    if va is not None and vb is not None:
                        acc[t].append(va - vb)
            print(f'\n  부트스트랩 (패킷 재표집 + 잡음 재추출, {nboot} 복제): D1 as-run - 표준RX(정렬)')
            for t in THR:
                if acc[t]:
                    v = np.array(acc[t])
                    lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
                    print(f'    {t:.0f}% 임계: {np.median(v):+6.2f} dB  [{lo:+6.2f}, {hi:+6.2f}]  '
                          f'(n={len(v)}, 비대칭비 {(hi-np.median(v))/max(np.median(v)-lo,1e-9):.1f}x)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--sf', type=int, default=7)
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--proto-frac', type=float, default=0.33)
    ap.add_argument('--max-clean', type=int, default=30)
    ap.add_argument('--U', type=int, default=100)
    ap.add_argument('--boot', type=int, default=0)
    a = ap.parse_args()
    main(a.root, a.sf, a.n, a.proto_frac, a.max_clean, a.U, a.boot)
