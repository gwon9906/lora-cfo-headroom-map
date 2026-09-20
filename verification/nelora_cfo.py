# -*- coding: utf-8 -*-
"""
세 검출기에 '같은' 동기 보상을 걸고 다시 잰다.

nelora_sync.py 결론: 경험적 프로토타입과 이상적 chirp 의 |상관| 0.83 은
전 코드 공통의 상수 오프셋(tau=+2.0 샘플, eps=+0.19 bin) 이었다. 보정하면 0.99.
따라서 보정 없이 잰 D2->D3 +2.37 dB 는 '하드웨어 왜곡' 이 아니다.

보상 수준 세 가지를 같은 심볼에 걸어 비교한다:
  C0  보정 없음          (기존 결과)
  C1  전역 상수 보정     (프로토타입 패킷에서만 추정한 tau/eps 하나)
  C2  패킷별 preamble 보정 (실제 LoRa 수신기가 하는 일: 패킷 자기 preamble 로 추정)

C2 는 심볼을 보정한 뒤 경험적 프로토타입도 보정된 심볼로 다시 만든다.
그래야 세 검출기가 같은 좌표계에 놓인다.

사용법:
  python nelora_cfo.py <데이터폴더> [--sf 7] [--n 4000]
"""
import os, glob, argparse
import numpy as np

from nelora_mf_test import (ideal_chirps, norm_rows, d1_nelora, d_mf, add_awgn,
                            empirical_prototypes, parse_raw_name, load_raw, cross)
from nelora_sync import frac_shift, best_align, K

PREAMBLE = 8          # symbol_idx 0..7 = code 0 업چ프
TAUS = np.arange(-8, 8.01, 0.5)
EPS_MAX = 2.0


def apply_corr(Y, tau, eps):
    """추정한 (tau, eps) 를 되돌린다: 프로토타입 쪽이 아니라 신호 쪽을 정렬."""
    M = Y.shape[-1]
    n = np.arange(M)
    return frac_shift(Y, -tau)*np.exp(-2j*np.pi*eps*n/M)


def est_packet(Yp, p0):
    """패킷의 preamble 심볼들로 (tau, eps) 를 비동기 결합 추정."""
    M = Yp.shape[1]
    kmax = int(round(EPS_MAX*K))
    ks = np.concatenate([np.arange(kmax+1), np.arange(-kmax, 0)])/K
    best, btau, beps = -1.0, 0.0, 0.0
    for tau in TAUS:
        ps = frac_shift(p0, tau)
        ps = ps/(np.linalg.norm(ps) + 1e-30)
        g = Yp*np.conj(ps)[None, :]
        G = np.abs(np.fft.fft(g, M*K, axis=1))
        acc = np.concatenate([G[:, :kmax+1], G[:, -kmax:]], axis=1).sum(0)
        j = int(np.argmax(acc))
        if acc[j] > best:
            best, btau, beps = acc[j], tau, ks[j]
    return btau, beps


def evaluate(Y, lab, Pidn, Pemp, down, osf, snrs, rng, tag):
    rows = []
    print(f"\n  [{tag}]  {'SNR':>6}{'D1':>10}{'D2':>10}{'D3':>10}")
    for snr in [None] + list(snrs):
        Yn = Y if snr is None else add_awgn(Y, snr, rng)
        e1 = (d1_nelora(Yn, down, osf) != lab).mean()*100
        e2 = (d_mf(Yn, Pidn) != lab).mean()*100
        e3 = (d_mf(Yn, Pemp) != lab).mean()*100
        name = '원본' if snr is None else f'{snr:.0f}'
        print(f"  {'':>{len(tag)+4}}{name:>6}{e1:>9.1f}%{e2:>9.1f}%{e3:>9.1f}%")
        if snr is not None:
            rows.append((snr, e1, e2, e3))
    rows.sort()
    xs = [r[0] for r in rows]
    return [cross(xs, [r[k] for r in rows]) for k in (1, 2, 3)]


def main(root, sf_want, n_max, proto_pkts, max_clean, snrs):
    files = sorted(glob.glob(os.path.join(root, '**', '*.mat'), recursive=True))
    recs = [m for m in (parse_raw_name(f) for f in files) if m and m['sf'] == sf_want]
    Pid, osf, M, N = ideal_chirps(sf_want, 125e3)
    Pidn = norm_rows(Pid)

    pkts = sorted({r['packet'] for r in recs}, key=int)
    proto_set = set(pkts[:proto_pkts])

    # 패킷별로 preamble + payload 를 통째로 적재
    bypkt = {}
    for r in recs:
        bypkt.setdefault(r['packet'], []).append(r)
    rng = np.random.default_rng(0)

    print(f'SF{sf_want}  OSF={osf}  N={N}  M={M}  패킷 {len(pkts)}개')
    print('패킷별 preamble 로 (tau, eps) 추정 중...')
    corr = {}
    for q, lst in bypkt.items():
        pre = sorted([r for r in lst if r['symbol'] < PREAMBLE], key=lambda r: r['symbol'])
        Yp = np.array([load_raw(r['path']) for r in pre if load_raw(r['path']).size == M])
        corr[q] = est_packet(Yp, Pid[0]) if len(Yp) else (0.0, 0.0)
    tau_all = np.array([c[0] for c in corr.values()])
    eps_all = np.array([c[1] for c in corr.values()])
    print(f'  추정 tau: 중앙값 {np.median(tau_all):+.2f} 샘플, '
          f'IQR {np.percentile(tau_all,25):+.2f}~{np.percentile(tau_all,75):+.2f}')
    print(f'  추정 eps: 중앙값 {np.median(eps_all):+.3f} bin, '
          f'IQR {np.percentile(eps_all,25):+.3f}~{np.percentile(eps_all,75):+.3f}, '
          f'범위 {eps_all.min():+.3f}~{eps_all.max():+.3f}')
    print('  -> eps 가 패킷마다 흩어져 있으면 패킷별 CFO 가 실재한다는 뜻')

    # payload 심볼 적재 (원본 / C2 보정본 둘 다)
    def payload(pset):
        out_raw, out_fix, labs = [], [], []
        for q in pset:
            tau, eps = corr[q]
            lst = [r for r in bypkt[q] if r['symbol'] >= 12]
            A = []
            for r in lst:
                y = load_raw(r['path'])
                if y.size == M:
                    A.append(y); labs.append(r['code_label'] % N)
            if not A:
                continue
            A = np.array(A)
            out_raw.append(A); out_fix.append(apply_corr(A, tau, eps))
        return np.vstack(out_raw), np.vstack(out_fix), np.array(labs)

    Praw, Pfix, plab = payload([q for q in pkts if q in proto_set])
    Traw, Tfix, tlab = payload([q for q in pkts if q not in proto_set])
    if len(Traw) > n_max:
        sel = rng.choice(len(Traw), n_max, replace=False)
        Traw, Tfix, tlab = Traw[sel], Tfix[sel], tlab[sel]
    print(f'\n프로토타입 {len(Praw)}심볼 / 평가 {len(Traw)}심볼 (패킷 분리 유지)')

    def build(Pset, labs):
        by = {c: [] for c in range(N)}
        for y, c in zip(Pset, labs):
            if len(by[c]) < max_clean:
                by[c].append(y)
        return empirical_prototypes(by, N, M)

    Pemp0, f0, _ = build(Praw, plab)
    Pemp2, f2, _ = build(Pfix, plab)

    # --- C1: 프로토타입 집합에서만 추정한 전역 상수
    best, btau, beps = best_align(Pemp0, Pid, TAUS, EPS_MAX)
    ok = np.isfinite(f0)
    g_tau = float(np.median(btau[ok])); g_eps = float(np.median(beps[ok]))
    print(f'\nC1 전역 상수 (프로토타입 집합에서만 추정): tau={g_tau:+.2f} 샘플, eps={g_eps:+.3f} bin')
    n = np.arange(M)
    Pid_al = norm_rows(frac_shift(Pid, g_tau)*np.exp(2j*np.pi*g_eps*n/M)[None, :])
    print(f'   보정 후 |상관| 평균 {np.abs(np.sum(Pemp0*np.conj(Pid_al),1))[ok].mean():.4f}')
    print(f'C2 패킷별 보정 후 프로토타입 일관성(1st 특이값 비중) '
          f'{np.nanmean(f0):.3f} -> {np.nanmean(f2):.3f}')
    simfix = np.abs(np.sum(Pemp2*np.conj(Pidn), 1))
    print(f'   C2 프로토타입 <-> 이상 chirp |상관| {simfix[np.isfinite(f2)].mean():.4f}')

    print('\n' + '='*66)
    res = {}
    res['C0 보정 없음'] = evaluate(Traw, tlab, Pidn, Pemp0,
                                np.conj(Pid[0]), osf, snrs, np.random.default_rng(1), 'C0')
    res['C1 전역 상수'] = evaluate(Traw, tlab, Pid_al, Pemp0,
                                np.conj(Pid_al[0]), osf, snrs, np.random.default_rng(1), 'C1')
    res['C2 패킷 preamble'] = evaluate(Tfix, tlab, Pidn, Pemp2,
                                     np.conj(Pid[0]), osf, snrs, np.random.default_rng(1), 'C2')

    print('\n' + '='*66)
    print('10% SER 도달 SNR (합성 AWGN 기준) 과 검출기 간 격차')
    print(f"  {'보상':<18}{'D1':>9}{'D2':>9}{'D3':>9} |{'D1->D2':>9}{'D2->D3':>9}{'D1->D3':>9}")
    for nm, c in res.items():
        f = lambda v: '  없음  ' if v is None else f'{v:7.2f}'
        g = lambda a, b: '    -   ' if (a is None or b is None) else f'{a-b:+7.2f}'
        print(f'  {nm:<18}{f(c[0]):>9}{f(c[1]):>9}{f(c[2]):>9} |'
              f'{g(c[0],c[1]):>9}{g(c[1],c[2]):>9}{g(c[0],c[2]):>9}')
    print('\n  참고: 논문이 신경망으로 보고한 이득 = 1.84 ~ 2.35 dB')
    print('  C2 에서 D2->D3 가 0 에 가까워지면, 원래의 +2.37 dB 는 동기 오차였다는 결론이 확정된다.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--sf', type=int, default=7)
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--proto-pkts', type=int, default=60)
    ap.add_argument('--max-clean', type=int, default=30)
    ap.add_argument('--snrs', type=str, default='0,-5,-10,-12,-14,-16,-18,-20')
    a = ap.parse_args()
    main(a.root, a.sf, a.n, a.proto_pkts, a.max_clean,
         [float(x) for x in a.snrs.split(',')])
