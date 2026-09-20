# -*- coding: utf-8 -*-
"""
SF7/8/9/10 에 같은 검사를 돌린다.

SF7 에서 나온 결론이 SF 를 올려도 유지되는가:
  - 전역 상수 오프셋(tau, eps)이 존재하는가
  - 보정하면 D2->D3 격차가 사라지는가
  - D1(dechirp+abs alias) 의 손실은 얼마나 남는가

주의: 고SF 는 코드 수가 2^SF 로 늘지만 캡처 수는 그대로여서 코드당 캡처가 급감한다.
      D3(경험적 MF)는 코드별 프로토타입이 필요하므로 SF9/10 에서는 성립하지 않는다.
      그래서 커버리지를 함께 찍고, 부족하면 D3 을 '참고값' 으로만 본다.

정렬 추정도 프로토타입 경유가 아니라 '라벨이 붙은 심볼에서 직접' 한다:
  max_{tau,eps} sum_i |<y_i, p_id[c_i](tau,eps)>|  (c_i 는 그 심볼의 라벨)
이러면 코드당 캡처가 부족해도 전역 오프셋은 잡을 수 있다.

사용법:
  python nelora_allsf.py <데이터셋루트> [--sfs 7,8,9,10]
"""
import os, glob, argparse
import numpy as np

from nelora_mf_test import (ideal_chirps, norm_rows, d_mf, add_awgn,
                            empirical_prototypes, parse_raw_name, load_raw, cross)
from nelora_sync import frac_shift
from nelora_fair import d1_abs_alias

# SF 가 1 오르면 처리이득이 약 3 dB 늘어 곡선이 통째로 내려간다.
# 고정 격자를 쓰면 고SF 에서 10% 를 만나지 못해 '교차 없음' 이 된다.
def snr_grid(sf):
    return list(np.arange(-8, -34, -2) - 3*(sf - 7))

TAUS = np.arange(-8, 8.01, 0.5)
EPS = np.arange(-1, 1.001, 1/16)


def align_from_labels(Y, lab, Pid, n_use=300, rng=None):
    """라벨을 아는 심볼들로 전역 (tau, eps) 추정. 코드별 프로토타입이 필요 없다."""
    M = Pid.shape[1]
    rng = rng or np.random.default_rng(0)
    if len(Y) > n_use:
        s = rng.choice(len(Y), n_use, replace=False)
        Y, lab = Y[s], lab[s]
    n = np.arange(M)
    E = np.exp(-2j*np.pi*EPS[None, :]*n[:, None]/M)      # (M, n_eps)
    best, btau, beps = -1.0, 0.0, 0.0
    for tau in TAUS:
        Ps = norm_rows(frac_shift(Pid, tau))
        g = Y*np.conj(Ps[lab])                            # (n_sym, M)
        sc = np.abs(g @ E).sum(0)                         # (n_eps,)
        j = int(np.argmax(sc))
        if sc[j] > best:
            best, btau, beps = sc[j], tau, EPS[j]
    return btau, beps


def ser_cross(fn, Y, lab, snrs, seed=1):
    rng = np.random.default_rng(seed)
    rows = [(s, (fn(add_awgn(Y, s, rng)) != lab).mean()*100) for s in snrs]
    rows.sort()
    return cross([r[0] for r in rows], [r[1] for r in rows]), dict(rows)


def run_sf(root, sf, n_test, proto_frac, max_clean):
    n_test = max(1000, int(n_test*(2.0**(7 - sf))**0.5))   # M 이 커지면 표본을 줄인다
    files = sorted(glob.glob(os.path.join(root, str(sf), '**', '*.mat'), recursive=True))
    recs = [m for m in (parse_raw_name(f) for f in files) if m and m['sf'] == sf]
    if not recs:
        print(f'SF{sf}: 파일 없음'); return None
    Pid, osf, M, N = ideal_chirps(sf, 125e3)
    Pidn = norm_rows(Pid)
    nn = np.arange(M)

    bypkt = {}
    for r in recs:
        bypkt.setdefault(r['packet'], []).append(r)
    pkts = sorted(bypkt, key=int)
    npro = max(1, int(round(len(pkts)*proto_frac)))
    proto_set = set(pkts[:npro])

    def payload(pset, cap=None):
        A, L = [], []
        for q in pset:
            for r in [x for x in bypkt[q] if x['symbol'] >= 12]:
                y = load_raw(r['path'])
                if y.size == M:
                    A.append(y.astype(np.complex64)); L.append(r['code_label'] % N)
            if cap and len(A) >= cap:
                break
        return np.array(A, dtype=np.complex64), np.array(L)

    Praw, plab = payload([q for q in pkts if q in proto_set])
    Traw, tlab = payload([q for q in pkts if q not in proto_set], cap=n_test*2)
    rng = np.random.default_rng(0)
    if len(Traw) > n_test:
        s = rng.choice(len(Traw), n_test, replace=False)
        Traw, tlab = Traw[s], tlab[s]

    # 코드 커버리지
    cov = np.bincount(plab, minlength=N)
    print(f'\nSF{sf}  N={N}  M={M}  패킷 {len(pkts)}개  '
          f'프로토타입 {npro}패킷/{len(Praw)}심볼  평가 {len(Traw)}심볼')
    print(f'  코드 커버리지: {int((cov>0).sum())}/{N} 코드, 코드당 중앙값 {int(np.median(cov))}개'
          f'  -> D3 {"성립" if (cov>0).mean() > 0.95 and np.median(cov) >= 5 else "**부족(참고값)**"}')

    g_tau, g_eps = align_from_labels(Praw, plab, Pid)
    print(f'  전역 정렬 추정(프로토타입 패킷에서만): tau={g_tau:+.2f} 샘플, eps={g_eps:+.3f} bin')
    Pid_al = norm_rows(frac_shift(Pid, g_tau)*np.exp(2j*np.pi*g_eps*nn/M)[None, :])

    by = {c: [] for c in range(N)}
    for y, c in zip(Praw, plab):
        if len(by[c]) < max_clean:
            by[c].append(y.astype(np.complex128))
    Pemp, f0, _ = empirical_prototypes(by, N, M)
    ok = np.isfinite(f0)
    sim0 = np.abs(np.sum(Pemp[ok]*np.conj(Pidn[ok]), 1)).mean()
    sim1 = np.abs(np.sum(Pemp[ok]*np.conj(Pid_al[ok]), 1)).mean()
    print(f'  경험 프로토타입 <-> 이상 chirp |상관|: 정렬 전 {sim0:.4f} -> 정렬 후 {sim1:.4f}')

    grid = snr_grid(sf)
    d3_ok = (cov > 0).mean() > 0.95 and np.median(cov) >= 5
    print(f'  SNR 격자 {grid[0]:.0f} ~ {grid[-1]:.0f} dB'
          + ('' if d3_ok else '   (D3 은 커버리지 부족으로 계산 생략)'))
    out = {}
    for tag, Pi, dn in (('C0 보정 없음', Pidn, np.conj(Pid[0])),
                        ('C1 전역 정렬', Pid_al, np.conj(Pid_al[0]))):
        c1, _ = ser_cross(lambda Y: d1_abs_alias(Y, dn, osf), Traw, tlab, grid)
        c2, _ = ser_cross(lambda Y: d_mf(Y, Pi), Traw, tlab, grid)
        c3 = ser_cross(lambda Y: d_mf(Y, Pemp), Traw, tlab, grid)[0] if d3_ok else None
        out[tag] = (c1, c2, c3)
    return dict(sf=sf, N=N, cov=(cov > 0).mean(), covmed=int(np.median(cov)),
                tau=g_tau, eps=g_eps, sim0=sim0, sim1=sim1, res=out)


def main(root, sfs, n_test, proto_frac, max_clean):
    allr = []
    for sf in sfs:
        r = run_sf(root, sf, n_test, proto_frac, max_clean)
        if r:
            allr.append(r)

    print('\n' + '='*78)
    print('요약: 10% SER 도달 SNR (합성 AWGN) 과 검출기 간 격차')
    f = lambda v: '  없음 ' if v is None else f'{v:6.2f}'
    g = lambda a, b: '   -  ' if (a is None or b is None) else f'{a-b:+6.2f}'
    print(f"  {'SF':>3}{'보상':<14}{'D1':>8}{'D2':>8}{'D3':>8} |"
          f"{'D1->D2':>8}{'D2->D3':>8}  D3 신뢰")
    for r in allr:
        for tag, (c1, c2, c3) in r['res'].items():
            trust = 'ok' if r['cov'] > 0.95 and r['covmed'] >= 5 else '부족'
            print(f"  {r['sf']:>3}{tag:<14}{f(c1):>8}{f(c2):>8}{f(c3):>8} |"
                  f"{g(c1,c2):>8}{g(c2,c3):>8}  {trust}")
    print('\n  전역 오프셋과 정렬 효과')
    print(f"  {'SF':>3}{'tau(샘플)':>11}{'eps(bin)':>10}{'|상관| 전':>11}{'|상관| 후':>11}"
          f"{'코드커버리지':>13}")
    for r in allr:
        print(f"  {r['sf']:>3}{r['tau']:>11.2f}{r['eps']:>10.3f}{r['sim0']:>11.4f}"
              f"{r['sim1']:>11.4f}{r['cov']*100:>11.0f}%  (중앙 {r['covmed']}개)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--sfs', type=str, default='7,8,9,10')
    ap.add_argument('--n', type=int, default=2500)
    ap.add_argument('--proto-frac', type=float, default=0.33)
    ap.add_argument('--max-clean', type=int, default=30)
    a = ap.parse_args()
    main(a.root, [int(x) for x in a.sfs.split(',')], a.n, a.proto_frac, a.max_clean)
