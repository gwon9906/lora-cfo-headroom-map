# -*- coding: utf-8 -*-
"""
D2->D3 격차가 '하드웨어 왜곡' 인가 '동기 오차' 인가.

A. 패킷별 오프셋: 검출기 예측이 라벨과 어긋나는 방향이 패킷 안에서 일관된가.
     일관된 +-1 bin 이면 패킷마다 걸린 CFO/타이밍이다.
B. 상관 탐색: 경험적 프로토타입을 (타이밍 tau, CFO eps) 로 맞춰 보정했을 때
     이상적 chirp 와의 |상관| 이 0.83 에서 얼마나 올라가는가.
       크게 오른다 -> D2->D3 는 동기 오차였다 (하드웨어 왜곡 아님)
       그대로다    -> 진짜 파형 부정합이다

  <p_emp, p_id(tau) * exp(j2pi eps n/M)> 는 g = p_emp * conj(p_id(tau)) 의
  DFT 이므로, eps 축 전체를 zero-padded FFT 한 번으로 훑는다.

사용법:
  python nelora_sync.py <데이터폴더> [--sf 7]
"""
import os, glob, argparse
from collections import Counter, defaultdict
import numpy as np

from nelora_mf_test import (ideal_chirps, norm_rows, d1_nelora, d_mf,
                            empirical_prototypes, parse_raw_name, load_raw)

K = 16          # CFO 격자 세밀도: 1/K bin


def frac_shift(X, tau):
    """주파수영역 위상 경사로 분수 샘플 타이밍 이동."""
    M = X.shape[-1]
    f = np.fft.fftfreq(M)
    return np.fft.ifft(np.fft.fft(X, axis=-1)*np.exp(-2j*np.pi*f*tau), axis=-1)


def best_align(Pemp, Pid, taus, eps_max):
    """코드별 max_{tau,eps} |<p_emp, p_id 를 tau/eps 로 민 것>| 과 그 인자."""
    N, M = Pemp.shape
    kmax = int(round(eps_max*K))
    best = np.zeros(N); btau = np.zeros(N); beps = np.zeros(N)
    for tau in taus:
        Ps = norm_rows(frac_shift(Pid, tau))
        g = Pemp*np.conj(Ps)                       # (N, M)
        G = np.abs(np.fft.fft(g, M*K, axis=1))     # eps = k/K bin
        if kmax == 0:                              # eps 고정: -0: 슬라이스가 전체를 집는다
            cand, ks = G[:, :1], np.zeros(1)
        else:
            cand = np.concatenate([G[:, :kmax+1], G[:, -kmax:]], axis=1)
            ks = np.concatenate([np.arange(kmax+1), np.arange(-kmax, 0)])/K
        j = np.argmax(cand, 1)
        v = cand[np.arange(N), j]
        upd = v > best
        best[upd] = v[upd]; btau[upd] = tau; beps[upd] = ks[j][upd]
    return best, btau, beps


def main(root, sf_want, n_max, proto_pkts, max_clean):
    files = sorted(glob.glob(os.path.join(root, '**', '*.mat'), recursive=True))
    recs = [m for m in (parse_raw_name(f) for f in files) if m and m['sf'] == sf_want]
    Pid, osf, M, N = ideal_chirps(sf_want, 125e3)
    down = np.conj(Pid[0]); Pidn = norm_rows(Pid)

    pkts = sorted({r['packet'] for r in recs}, key=int)
    proto_set = set(pkts[:proto_pkts])
    proto = [r for r in recs if r['packet'] in proto_set and r['symbol'] >= 12]
    test  = [r for r in recs if r['packet'] not in proto_set and r['symbol'] >= 12]

    by = {c: [] for c in range(N)}
    for r in proto:
        c = r['code_label'] % N
        if len(by[c]) < max_clean:
            y = load_raw(r['path'])
            if y.size == M:
                by[c].append(y)
    Pemp, frac, cnt = empirical_prototypes(by, N, M)

    rng = np.random.default_rng(0)
    if len(test) > n_max:
        test = [test[i] for i in rng.choice(len(test), n_max, replace=False)]
    Y, lab, pkt = [], [], []
    for r in test:
        y = load_raw(r['path'])
        if y.size == M:
            Y.append(y); lab.append(r['code_label'] % N); pkt.append(r['packet'])
    Y = np.array(Y); lab = np.array(lab); pkt = np.array(pkt)

    # ------------------------------------------------------------------ A
    print('='*72)
    print('A  패킷별 오프셋이 일관된 방향인가  (합성잡음 없음, 캡처 원본)')
    preds = {'D1': d1_nelora(Y, down, osf), 'D2': d_mf(Y, Pidn), 'D3': d_mf(Y, Pemp)}
    for nm, p in preds.items():
        d = (p - lab) % N
        d = np.where(d > N//2, d - N, d)
        print(f'\n  [{nm}]  전체 오프셋 분포: ' +
              ', '.join(f'{k:+d}:{v}' for k, v in sorted(Counter(d).items())[:9]))
        per = defaultdict(list)
        for q, dd in zip(pkt, d):
            per[q].append(dd)
        # 패킷마다 '가장 흔한 오프셋' 이 0 이 아닌 패킷 = 통째로 밀린 패킷
        shifted, strong = [], []
        for q, v in per.items():
            c = Counter(v).most_common(1)[0]
            if c[0] != 0:
                shifted.append(q)
                if c[1]/len(v) >= 0.5:
                    strong.append((q, c[0], c[1], len(v)))
        print(f'    최빈 오프셋이 0 이 아닌 패킷: {len(shifted)}/{len(per)}개'
              f'  (그중 과반이 같은 값: {len(strong)}개)')
        if strong:
            s = sorted(strong, key=lambda x: -x[2]/x[3])[:6]
            print('    예: ' + ', '.join(f'pkt{q} {o:+d}bin {c}/{t}' for q, o, c, t in s))
        # 패킷 내부 일관성: 오프셋의 패킷별 최빈값 비율
        hom = np.mean([Counter(v).most_common(1)[0][1]/len(v) for v in per.values()])
        print(f'    패킷 내부 오프셋 동질성(최빈값 점유율) 평균 {hom*100:.1f}%')

    # ------------------------------------------------------------------ B
    print('\n' + '='*72)
    print('B  경험적 프로토타입 <-> 이상적 chirp 상관, 동기 보정 전/후')
    ok = np.isfinite(frac)
    base = np.abs(np.sum(Pemp*np.conj(Pidn), 1))
    print(f'  보정 전 |상관| 평균 {base[ok].mean():.4f}  (중앙값 {np.median(base[ok]):.4f})')

    for label, taus, emax in (('타이밍만 (eps=0)', np.arange(-2*osf, 2*osf+0.5, 0.5), 0.0),
                              ('CFO만 (tau=0)',   np.array([0.0]),                    2.0),
                              ('타이밍+CFO',      np.arange(-2*osf, 2*osf+0.5, 0.5), 2.0)):
        best, btau, beps = best_align(Pemp, Pid, taus, emax)
        print(f'\n  [{label}]  |상관| 평균 {best[ok].mean():.4f} '
              f'(중앙값 {np.median(best[ok]):.4f})')
        if len(taus) > 1:
            t = btau[ok]
            print(f'    최적 tau: 중앙값 {np.median(t):+.2f} 샘플, '
                  f'IQR {np.percentile(t,25):+.2f}~{np.percentile(t,75):+.2f}, '
                  f'최빈 {Counter(t).most_common(3)}')
        if emax > 0:
            e = beps[ok]
            print(f'    최적 eps: 중앙값 {np.median(e):+.3f} bin, '
                  f'IQR {np.percentile(e,25):+.3f}~{np.percentile(e,75):+.3f}')

    print('\n  해석: 평균 |상관| 이 1 에 가깝게 올라가면 0.83 은 동기 오차였다는 뜻이고,')
    print('        D2->D3 의 +2.37 dB 는 하드웨어 왜곡이 아니라 미보상 CFO/타이밍이다.')
    print('        최적 tau/eps 가 코드마다 제각각이면 공통 오프셋이 아니라 잔차이므로 해석이 다르다.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--sf', type=int, default=7)
    ap.add_argument('--n', type=int, default=6000)
    ap.add_argument('--proto-pkts', type=int, default=60)
    ap.add_argument('--max-clean', type=int, default=30)
    a = ap.parse_args()
    main(a.root, a.sf, a.n, a.proto_pkts, a.max_clean)
