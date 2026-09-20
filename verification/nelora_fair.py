# -*- coding: utf-8 -*-
"""
헤드라인 숫자를 쓰기 전에 남은 두 구멍을 막는다.

F1  통째로 밀린 패킷(147, 84, 177)의 정체.
      CFO 로는 안 고쳐졌다. 심볼 추출 창이 정수 bin(=OSF 샘플) 만큼 밀린 것인가?
      그렇다면 데이터셋 라벨/추출 버그이지 검출기 문제가 아니다.

F2  D1 을 제대로 구현했을 때도 4 dB 를 잃는가.
      이론상 LoRa 의 dechirp+FFT 는 '데시메이션을 제대로 하면' 정합필터와 같다.
      NELoRa 의 chirp_abs_alias 는 OSF 배 오버샘플 상태에서 alias 두 조각을
      abs 로 더하는 근사다. 브릭월 데시메이션 판과 비교해서, 격차가 남는지 본다.
      남지 않으면 'D1->D2 4.3 dB' 는 NELoRa 구현 선택의 손실이지 알고리즘 한계가 아니다.

사용법:
  python nelora_fair.py <데이터폴더> [--sf 7]
"""
import os, glob, argparse
import numpy as np

from nelora_mf_test import (ideal_chirps, norm_rows, d_mf, add_awgn,
                            empirical_prototypes, parse_raw_name, load_raw, cross)
from nelora_sync import frac_shift, best_align
from nelora_cfo import TAUS, EPS_MAX

SNRS = [0, -5, -10, -12, -14, -16, -18, -20]


def d1_abs_alias(Y, down, osf):
    """NELoRa 원본: 오버샘플 FFT 의 양끝 조각을 abs 로 더한다."""
    M = Y.shape[1]
    Z = np.fft.fft(Y*down[None, :], M, axis=1)
    T = int(round(M/osf))
    return np.argmax(np.abs(Z[:, :T]) + np.abs(Z[:, -T:]), 1) % T


def d1_decimate(Y, down, osf):
    """교과서적 LoRa 수신기: dechirp -> 브릭월 저역통과 -> N 샘플 데시메이션 -> FFT."""
    M = Y.shape[1]
    N = M//osf
    Z = np.fft.fft(Y*down[None, :], M, axis=1)
    # |f| < BW/2 인 성분만 남긴다 = 앞 N/2, 뒤 N/2 빈
    Zd = np.concatenate([Z[:, :N//2], Z[:, -(N - N//2):]], axis=1)
    return np.argmax(np.abs(Zd), 1) % N


def ser(fn, Y, lab, seed=1):
    rng = np.random.default_rng(seed)
    rows = [(s, (fn(add_awgn(Y, s, rng)) != lab).mean()*100) for s in SNRS]
    rows.sort()
    return cross([r[0] for r in rows], [r[1] for r in rows]), dict(rows)


def main(root, sf_want, n_max, proto_pkts, max_clean):
    files = sorted(glob.glob(os.path.join(root, '**', '*.mat'), recursive=True))
    recs = [m for m in (parse_raw_name(f) for f in files) if m and m['sf'] == sf_want]
    Pid, osf, M, N = ideal_chirps(sf_want, 125e3)
    Pidn = norm_rows(Pid)
    nn = np.arange(M)

    bypkt = {}
    for r in recs:
        bypkt.setdefault(r['packet'], []).append(r)
    pkts = sorted(bypkt, key=int)
    proto_set = set(pkts[:proto_pkts])

    def payload(pset):
        A, L, Q = [], [], []
        for q in pset:
            for r in [x for x in bypkt[q] if x['symbol'] >= 12]:
                y = load_raw(r['path'])
                if y.size == M:
                    A.append(y); L.append(r['code_label'] % N); Q.append(q)
        return np.array(A), np.array(L), np.array(Q)

    Praw, plab, _ = payload([q for q in pkts if q in proto_set])
    Traw, tlab, tpkt = payload([q for q in pkts if q not in proto_set])

    by = {c: [] for c in range(N)}
    for y, c in zip(Praw, plab):
        if len(by[c]) < max_clean:
            by[c].append(y)
    Pemp, f0, _ = empirical_prototypes(by, N, M)
    ok = np.isfinite(f0)
    _, btau, beps = best_align(Pemp, Pid, TAUS, EPS_MAX)
    g_tau = float(np.median(btau[ok])); g_eps = float(np.median(beps[ok]))
    Pid_al = norm_rows(frac_shift(Pid, g_tau)*np.exp(2j*np.pi*g_eps*nn/M)[None, :])
    down_al = np.conj(Pid_al[0])
    print(f'SF{sf_want}  전역 정렬 tau={g_tau:+.2f} 샘플, eps={g_eps:+.3f} bin')

    # ------------------------------------------------------------------ F1
    print('\n' + '='*70)
    print('F1  통째로 밀린 패킷의 정체: 심볼 추출 창이 정수 bin 만큼 밀렸는가')
    print(f"  {'패킷':>6}{'n':>5}" + ''.join(f'{d:+6d}bin' for d in range(-2, 3)))
    for q in ['147', '84', '177', pkts[5], pkts[6]]:
        m = tpkt == q
        if not m.any():
            continue
        Yq, Lq = Traw[m], tlab[m]
        cells = []
        for d in range(-2, 3):
            # 프로토타입을 d bin (= d*osf 샘플) 만큼 민 것과 비교
            Pd = norm_rows(frac_shift(Pid_al, d*osf))
            cells.append((d_mf(Yq, Pd) != Lq).mean()*100)
        print(f'  {q:>6}{m.sum():>5}' + ''.join(f'{v:9.1f}%' for v in cells))
    print('  -> 어떤 d 에서 오류율이 급락하면 그 패킷은 d bin 만큼 밀려 잘린 것이다.')
    print('     (1 bin = OSF = %d 샘플. CFO 가 아니라 추출/라벨 정렬 문제.)' % osf)

    # ------------------------------------------------------------------ F2
    print('\n' + '='*70)
    print('F2  D1 을 제대로 데시메이션해서 구현하면 격차가 남는가')
    if len(Traw) > n_max:
        sel = np.random.default_rng(0).choice(len(Traw), n_max, replace=False)
        Ys, Ls = Traw[sel], tlab[sel]
    else:
        Ys, Ls = Traw, tlab
    arms = [
        ('D1 abs-alias (NELoRa 원본)', lambda Y: d1_abs_alias(Y, down_al, osf)),
        ('D1 브릭월 데시메이션',        lambda Y: d1_decimate(Y, down_al, osf)),
        ('D2 정합필터 (이상, 정렬)',    lambda Y: d_mf(Y, Pid_al)),
        ('D3 정합필터 (경험)',          lambda Y: d_mf(Y, Pemp)),
    ]
    print(f"  {'검출기':<28}{'10% SER':>10}{'-14dB':>9}{'-16dB':>9}{'-18dB':>9}")
    out = {}
    for nm, fn in arms:
        c, d = ser(fn, Ys, Ls)
        out[nm] = c
        print(f'  {nm:<28}{("없음" if c is None else f"{c:7.2f}dB"):>10}'
              f'{d[-14]:8.1f}%{d[-16]:8.1f}%{d[-18]:8.1f}%')
    a = out['D1 abs-alias (NELoRa 원본)']; b = out['D1 브릭월 데시메이션']
    c2 = out['D2 정합필터 (이상, 정렬)']
    if a and b:
        print(f'\n  abs-alias -> 데시메이션 : {a-b:+.2f} dB  (NELoRa 구현 선택의 손실)')
    if b and c2:
        print(f'  데시메이션 -> 정합필터  : {b-c2:+.2f} dB  (남으면 알고리즘 한계, 0 이면 동치)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--sf', type=int, default=7)
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--proto-pkts', type=int, default=60)
    ap.add_argument('--max-clean', type=int, default=30)
    a = ap.parse_args()
    main(a.root, a.sf, a.n, a.proto_pkts, a.max_clean)
