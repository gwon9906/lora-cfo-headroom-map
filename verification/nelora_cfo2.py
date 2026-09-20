# -*- coding: utf-8 -*-
"""
3순위 후속 검증 세 가지.

R1  C2 가 실패한 이유가 delay-Doppler 축퇴인가.
      선형 chirp 은 tau 이동과 eps 이동이 거의 같은 효과를 낸다. preamble 은
      전부 code 0 (같은 chirp) 이므로 2D 지표가 능선(ridge) 이 되어 분리 불가.
      128개 코드를 쓰는 C1 은 코드마다 순환이동이 달라 축퇴가 깨진다.
      + 이 데이터셋에 SFD 다운چ프(symbol_idx 10,11)가 있는지 확인.

R2  C3: tau 는 전역 상수로 고정하고 패킷별 eps 만 preamble 로 추정.
      축퇴를 피하면서 패킷별 CFO 를 잡는다. 통째로 밀린 패킷이 고쳐지는가.

R3  D1 이 억울하게 불리한 것은 아닌가.
      FFT 업샘플 U 와 alias 결합 방식(비동기 abs 합 vs 동기 합) 을 바꿔 본다.
      D1->D2 의 4.3 dB 가 구현 디테일이면 헤드라인을 그렇게 쓰면 안 된다.

사용법:
  python nelora_cfo2.py <데이터폴더> [--sf 7]
"""
import os, glob, argparse
from collections import Counter
import numpy as np

from nelora_mf_test import (ideal_chirps, norm_rows, d_mf, add_awgn,
                            empirical_prototypes, parse_raw_name, load_raw, cross)
from nelora_sync import frac_shift, best_align, K
from nelora_cfo import apply_corr, TAUS, EPS_MAX, PREAMBLE

SNRS = [0, -5, -10, -12, -14, -16, -18, -20]


def d1_variant(Y, down, osf, U=1, coherent=False):
    """NELoRa dechirp 계열. coherent=True 면 alias 를 복소합(동기)으로 결합."""
    M = Y.shape[1]
    Z = np.fft.fft(Y*down[None, :], M*U, axis=1)
    T = int(round(M*U/osf))
    s = np.abs(Z[:, :T] + Z[:, -T:]) if coherent else np.abs(Z[:, :T]) + np.abs(Z[:, -T:])
    return (np.round(np.argmax(s, 1)/U).astype(int)) % (M//osf)


def ser_curve(fn, Y, lab, rng_seed=1):
    rng = np.random.default_rng(rng_seed)
    rows = []
    for snr in SNRS:
        rows.append((snr, (fn(add_awgn(Y, snr, rng)) != lab).mean()*100))
    rows.sort()
    return cross([r[0] for r in rows], [r[1] for r in rows]), rows


def est_eps_only(Yp, p0, tau, emax=2.0):
    """tau 를 고정하고 eps 만 추정 (축퇴 회피)."""
    M = Yp.shape[1]
    kmax = int(round(emax*K))
    ks = np.concatenate([np.arange(kmax+1), np.arange(-kmax, 0)])/K
    ps = frac_shift(p0, tau); ps = ps/(np.linalg.norm(ps) + 1e-30)
    G = np.abs(np.fft.fft(Yp*np.conj(ps)[None, :], M*K, axis=1))
    acc = np.concatenate([G[:, :kmax+1], G[:, -kmax:]], axis=1).sum(0)
    return float(ks[int(np.argmax(acc))])


def main(root, sf_want, n_max, proto_pkts, max_clean):
    files = sorted(glob.glob(os.path.join(root, '**', '*.mat'), recursive=True))
    recs = [m for m in (parse_raw_name(f) for f in files) if m and m['sf'] == sf_want]
    Pid, osf, M, N = ideal_chirps(sf_want, 125e3)
    Pidn = norm_rows(Pid)
    n = np.arange(M)

    # ---------------------------------------------------------------- R1
    print('='*70)
    print('R1  delay-Doppler 축퇴 확인')
    sidx = Counter(r['symbol'] for r in recs)
    present = sorted(sidx)
    print(f'  존재하는 symbol_idx: {present[:14]} ... {present[-3:]}')
    missing = [i for i in range(max(present)+1) if i not in sidx]
    print(f'  빠진 symbol_idx: {missing}')
    print('  -> LoRa 프레임에서 8,9 는 sync word, 10,11 은 SFD 다운چ프 자리.')
    print(f'     {missing} 가 비어 있으면 타이밍/CFO 를 분리할 다운چ프가 없다.')

    # 한 패킷의 preamble 로 2D 지표를 그려 능선인지 본다
    bypkt = {}
    for r in recs:
        bypkt.setdefault(r['packet'], []).append(r)
    q0 = sorted(bypkt, key=int)[0]
    pre = sorted([r for r in bypkt[q0] if r['symbol'] < PREAMBLE], key=lambda r: r['symbol'])
    Yp = np.array([load_raw(r['path']) for r in pre])
    kmax = int(round(EPS_MAX*K))
    ks = np.concatenate([np.arange(kmax+1), np.arange(-kmax, 0)])/K
    surf = []
    for tau in TAUS:
        ps = frac_shift(Pid[0], tau); ps = ps/np.linalg.norm(ps)
        G = np.abs(np.fft.fft(Yp*np.conj(ps)[None, :], M*K, axis=1))
        surf.append(np.concatenate([G[:, :kmax+1], G[:, -kmax:]], axis=1).sum(0))
    surf = np.array(surf)                       # (tau, eps)
    surf /= surf.max()
    i, j = np.unravel_index(np.argmax(surf), surf.shape)
    print(f'\n  패킷 {q0} preamble 2D 지표: 최대점 tau={TAUS[i]:+.2f}, eps={ks[j]:+.3f}')
    near = surf > 0.99
    print(f'  최대값의 99% 이상인 (tau,eps) 점 개수: {near.sum()} / {surf.size}')
    if near.sum() > 1:
        ti, ej = np.where(near)
        print(f'    그 점들의 tau 범위 {TAUS[ti].min():+.2f}~{TAUS[ti].max():+.2f}, '
              f'eps 범위 {ks[ej].min():+.3f}~{ks[ej].max():+.3f}')
        if len(set(ti)) > 1:
            s = np.argsort(TAUS[ti])
            pts = list(zip(TAUS[ti][s], ks[ej][s]))[:6]
            print('    능선 위 점들: ' + ', '.join(f'({t:+.1f},{e:+.2f})' for t, e in pts))
            dt = TAUS[ti].max() - TAUS[ti].min()
            de = ks[ej].max() - ks[ej].min()
            if dt > 0:
                print(f'    능선 기울기 ~ {de/dt:+.4f} bin/샘플  '
                      f'(이론값 1/OSF = {1/osf:+.4f}: tau 1샘플 = eps 1/OSF bin)')
            print('  -> 능선이면 preamble 만으로는 tau 와 eps 를 분리할 수 없다. C2 실패의 원인.')
    else:
        print('  -> 뾰족한 최대점. 축퇴는 아니다.')

    # ---------------------------------------------------------------- 적재
    pkts = sorted(bypkt, key=int)
    proto_set = set(pkts[:proto_pkts])

    def payload(pset, fix=None):
        A, L, Q = [], [], []
        for q in pset:
            lst = [r for r in bypkt[q] if r['symbol'] >= 12]
            B = []
            for r in lst:
                y = load_raw(r['path'])
                if y.size == M:
                    B.append(y); L.append(r['code_label'] % N); Q.append(q)
            if not B:
                continue
            B = np.array(B)
            A.append(B if fix is None else apply_corr(B, *fix(q)))
        return np.vstack(A), np.array(L), np.array(Q)

    Praw, plab, _ = payload([q for q in pkts if q in proto_set])
    Traw, tlab, tpkt = payload([q for q in pkts if q not in proto_set])
    rng = np.random.default_rng(0)
    if len(Traw) > n_max:
        sel = rng.choice(len(Traw), n_max, replace=False)
        Traw, tlab, tpkt = Traw[sel], tlab[sel], tpkt[sel]

    def build(Pset, labs):
        by = {c: [] for c in range(N)}
        for y, c in zip(Pset, labs):
            if len(by[c]) < max_clean:
                by[c].append(y)
        return empirical_prototypes(by, N, M)

    Pemp0, f0, _ = build(Praw, plab)
    ok = np.isfinite(f0)
    best, btau, beps = best_align(Pemp0, Pid, TAUS, EPS_MAX)
    g_tau = float(np.median(btau[ok])); g_eps = float(np.median(beps[ok]))
    Pid_al = norm_rows(frac_shift(Pid, g_tau)*np.exp(2j*np.pi*g_eps*n/M)[None, :])
    print(f'\n전역 상수 (프로토타입 집합): tau={g_tau:+.2f}, eps={g_eps:+.3f}')

    # ---------------------------------------------------------------- R2
    print('\n' + '='*70)
    print('R2  C3: tau 전역 고정 + 패킷별 eps 만 preamble 로 추정')
    peps = {}
    for q in pkts:
        pr = sorted([r for r in bypkt[q] if r['symbol'] < PREAMBLE], key=lambda r: r['symbol'])
        Y8 = np.array([load_raw(r['path']) for r in pr if load_raw(r['path']).size == M])
        peps[q] = est_eps_only(Y8, Pid[0], g_tau, emax=0.5) if len(Y8) else 0.0
    ev = np.array([peps[q] for q in pkts])
    print(f'  패킷별 eps: 중앙값 {np.median(ev):+.3f} bin, '
          f'IQR {np.percentile(ev,25):+.3f}~{np.percentile(ev,75):+.3f}, '
          f'범위 {ev.min():+.3f}~{ev.max():+.3f}')
    print(f'  eps 가 0 이 아닌 패킷 {int((np.abs(ev) > 1e-9).sum())}/{len(ev)}개')
    for q in ('147', '84', '177'):
        if q in peps:
            print(f'    통째로 밀렸던 pkt{q}: eps={peps[q]:+.3f} bin')

    Pfix3, plab3, _ = payload([q for q in pkts if q in proto_set],
                              fix=lambda q: (g_tau, g_eps + peps[q]))
    Tfix3, tlab3, tpkt3 = payload([q for q in pkts if q not in proto_set],
                                  fix=lambda q: (g_tau, g_eps + peps[q]))
    if len(Tfix3) > n_max:
        Tfix3, tlab3, tpkt3 = Tfix3[sel], tlab3[sel], tpkt3[sel]
    Pemp3, f3, _ = build(Pfix3, plab3)
    print(f'  C3 프로토타입 일관성 {np.nanmean(f0):.3f} -> {np.nanmean(f3):.3f}')

    # 보정 전/후, 통째로 밀렸던 패킷의 오류율
    for nm, Yt, Pe, Pi in (('C1', Traw, Pemp0, Pid_al), ('C3', Tfix3, Pemp3, Pidn)):
        p = d_mf(Yt, Pi)
        bad = (p != tlab)
        sub = [(q, bad[tpkt3 == q].mean()*100) for q in ('147', '84', '177') if (tpkt3 == q).any()]
        print(f'  [{nm}] D2 전체 오류 {bad.mean()*100:.2f}%   ' +
              '  '.join(f'pkt{q} {v:.1f}%' for q, v in sub))

    # ---------------------------------------------------------------- R3
    print('\n' + '='*70)
    print('R3  D1 이 구현 디테일 때문에 불리한 것은 아닌가')
    down_al = np.conj(Pid_al[0])
    variants = [
        ('U=1 비동기 abs합 (원본)', lambda Y: d1_variant(Y, down_al, osf, 1, False)),
        ('U=2 비동기',             lambda Y: d1_variant(Y, down_al, osf, 2, False)),
        ('U=4 비동기',             lambda Y: d1_variant(Y, down_al, osf, 4, False)),
        ('U=1 동기(복소합)',       lambda Y: d1_variant(Y, down_al, osf, 1, True)),
        ('U=4 동기(복소합)',       lambda Y: d1_variant(Y, down_al, osf, 4, True)),
        ('D2 정합필터 (정렬됨)',   lambda Y: d_mf(Y, Pid_al)),
    ]
    print(f"  {'변형':<24}{'10% SER':>10}   SER@-16dB")
    for nm, fn in variants:
        c, rows = ser_curve(fn, Traw, tlab)
        at16 = dict((s, v) for s, v in rows).get(-16, float('nan'))
        print(f'  {nm:<24}{("없음" if c is None else f"{c:7.2f} dB"):>10}   {at16:5.1f}%')
    print('\n  -> U 나 결합방식을 바꿔도 D2 근처로 못 오면, D1->D2 격차는 실제 알고리즘 손실이다.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--sf', type=int, default=7)
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--proto-pkts', type=int, default=60)
    ap.add_argument('--max-clean', type=int, default=30)
    a = ap.parse_args()
    main(a.root, a.sf, a.n, a.proto_pkts, a.max_clean)
