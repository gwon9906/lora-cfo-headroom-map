# -*- coding: utf-8 -*-
"""
§6 재검증: 표준 LoRa 수신기(데시메이션 우선)를 제대로 돌린다.
동시에 §4 를 올바른 chirp 생성기로 다시 재고, 패킷 단위 부트스트랩 CI 를 붙인다.

바로잡은 것 두 가지:

  (1) 생성기.  nelora_mf_test.ideal_chirps 는 위상을 cumsum 으로 적분해서
      OSF=8 판과 OSF=1 판이 |상관| 0.71 밖에 안 됐다. 그래서 '표준 수신기' 를
      짜면 무잡음에서도 23% 밖에 못 맞췄다. nelora_chirp 의 해석적 생성기로 바꾼다.
      cumsum 판은 해석적 판과 정확히 (tau=-1.00 샘플, eps=-0.0625 bin) 차이이므로,
      §4 에서 잰 (+2.00, +0.188) 중 그만큼은 데이터가 아니라 내 생성기였다.

  (2) 순서.  이전 d1_decimate 는 de-chirp 를 먼저 하고 대역을 잘랐다. de-chirp 후의
      톤은 +-BW 에 걸쳐 있으므로 그건 신호를 자르는 짓이다. 표준 수신기는
      de-chirp *이전에* +-BW/2 로 거르고 N 샘플로 내린다. 그러면 대역 밖 잡음
      7/8 이 버려지고, 두 alias 조각이 같은 빈에 코히런트하게 접힌다.

사용법:
  python nelora_stdrx.py ideal [--sf 7]
  python nelora_stdrx.py real  <데이터폴더> [--sf 7] [--boot 1000]
"""
import os, glob, argparse
import numpy as np

from nelora_mf_test import (norm_rows, d_mf, add_awgn, empirical_prototypes,
                            parse_raw_name, load_raw, cross)
from nelora_chirp import chirp_bank, base_chirp
from nelora_sync import frac_shift
from nelora_cfo import apply_corr

SNRS = [-8, -10, -12, -14, -16, -18, -20, -22, -24]
TAUS = np.arange(-8, 8.01, 0.25)
EPS = np.arange(-1, 1.001, 1/32)


# ------------------------------------------------------------------ 검출기
def rx_standard(Y, base_n, osf, N, lpf=True):
    """표준 LoRa 수신기: (LPF ->) N 샘플 데시메이션 -> de-chirp -> N-FFT."""
    if lpf:
        Z = np.fft.fft(Y, axis=1)
        Zd = np.concatenate([Z[:, :N//2], Z[:, -(N - N//2):]], axis=1)
        yd = np.fft.ifft(Zd, axis=1)
    else:
        yd = Y[:, ::osf]
    return np.argmax(np.abs(np.fft.fft(yd*np.conj(base_n)[None, :], N, axis=1)), 1) % N


def rx_alias(Y, down, osf, N, nblk=2, mode='abs'):
    """OSF 상태에서 de-chirp 후 FFT 블록 결합 (NELoRa 계열)."""
    M = Y.shape[1]
    Z = np.fft.fft(Y*down[None, :], M, axis=1)
    if nblk == 2:
        blocks = [Z[:, :N], Z[:, -N:]]
    else:
        blocks = [Z[:, j*N:(j+1)*N] for j in range(nblk)]
    if mode == 'abs':
        s = sum(np.abs(b) for b in blocks)
    elif mode == 'comp':
        s = np.abs(sum(blocks))
    else:
        s = sum(np.abs(b)**2 for b in blocks)
    return np.argmax(s, 1) % N


def align_from_labels(Y, lab, P, n_use=400, seed=0):
    """라벨을 아는 심볼들로 전역 (tau, eps) 추정."""
    M = P.shape[1]
    rng = np.random.default_rng(seed)
    if len(Y) > n_use:
        s = rng.choice(len(Y), n_use, replace=False)
        Y, lab = Y[s], lab[s]
    n = np.arange(M)
    E = np.exp(-2j*np.pi*EPS[None, :]*n[:, None]/M)
    best, bt, be = -1.0, 0.0, 0.0
    for tau in TAUS:
        Ps = norm_rows(frac_shift(P, tau))
        sc = np.abs((Y*np.conj(Ps[lab])) @ E).sum(0)
        j = int(np.argmax(sc))
        if sc[j] > best:
            best, bt, be = sc[j], tau, EPS[j]
    return bt, be


# ------------------------------------------------------------------ 이상 신호
def cmd_ideal(sf, ntrial):
    X, osf, M, N = chirp_bank(sf, 8)
    Xn = norm_rows(X)
    base_n = base_chirp(sf, 1)
    down = np.conj(X[0])
    rng = np.random.default_rng(0)
    lab = rng.integers(0, N, ntrial)
    Y = X[lab]*np.exp(2j*np.pi*rng.random((ntrial, 1)))

    print('='*76)
    print(f'T1  이상 신호 온전성 (SF{sf}, {ntrial} trials, 타이밍/CFO = 0, 올바른 생성기)')
    print('    표준 수신기가 MF 와 붙어야 한다. 안 붙으면 아직 내 버그다.')
    arms = [
        ('MF (OSF=8 전대역 상관)',              lambda Y: d_mf(Y, Xn)),
        ('표준 수신기 (LPF->데시->dechirp)',     lambda Y: rx_standard(Y, base_n, osf, N, True)),
        ('표준 수신기 (LPF 없이 솎기만)',         lambda Y: rx_standard(Y, base_n, osf, N, False)),
        ('NELoRa abs-alias (2블록)',            lambda Y: rx_alias(Y, down, osf, N, 2, 'abs')),
    ]
    show(ser_table(arms, Y, lab))

    print('\n' + '='*76)
    print('T2  복소합 재확인: 블록 수와 결합 방식')
    arms2 = [
        ('2블록 abs (NELoRa 원본)', lambda Y: rx_alias(Y, down, osf, N, 2, 'abs')),
        ('2블록 복소합',            lambda Y: rx_alias(Y, down, osf, N, 2, 'comp')),
        ('2블록 전력합',            lambda Y: rx_alias(Y, down, osf, N, 2, 'pow')),
        (f'{osf}블록 abs',          lambda Y: rx_alias(Y, down, osf, N, osf, 'abs')),
        (f'{osf}블록 복소합',       lambda Y: rx_alias(Y, down, osf, N, osf, 'comp')),
        ('표준 수신기',             lambda Y: rx_standard(Y, base_n, osf, N, True)),
    ]
    show(ser_table(arms2, Y, lab))


def ser_table(arms, Y, lab, snrs=SNRS, seed=1):
    out = {}
    for nm, fn in arms:
        rng = np.random.default_rng(seed)
        rows = sorted((s, (fn(add_awgn(Y, s, rng)) != lab).mean()*100) for s in snrs)
        out[nm] = (cross([r[0] for r in rows], [r[1] for r in rows]), dict(rows))
    return out


def show(out, cols=(-14, -18, -20)):
    print(f"  {'검출기':<34}{'10% SER':>10}" + ''.join(f'{c:>8}dB' for c in cols))
    for nm, (c, d) in out.items():
        print(f'  {nm:<34}{("없음" if c is None else f"{c:7.2f}dB"):>10}'
              + ''.join(f'{d[c2]:9.1f}%' for c2 in cols))


# ------------------------------------------------------------------ 실데이터
def boot_cross(errs, pkt, snrs, nboot, seed=0):
    """패킷 단위 부트스트랩으로 10% SER 교차점의 CI. errs: {snr: bool array}."""
    rng = np.random.default_rng(seed)
    upk, inv = np.unique(pkt, return_inverse=True)
    idx_by = [np.where(inv == i)[0] for i in range(len(upk))]
    vals = []
    for _ in range(nboot):
        pick = rng.integers(0, len(upk), len(upk))
        sel = np.concatenate([idx_by[i] for i in pick])
        ys = [errs[s][sel].mean()*100 for s in snrs]
        v = cross(list(snrs), ys)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None, None
    v = np.array(vals)
    return float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def cmd_real(root, sf, n_test, proto_frac, max_clean, nboot):
    X, osf, M, N = chirp_bank(sf, 8)
    Xn = norm_rows(X)
    base_n = base_chirp(sf, 1)
    down = np.conj(X[0])

    files = sorted(glob.glob(os.path.join(root, '**', '*.mat'), recursive=True))
    recs = [m for m in (parse_raw_name(f) for f in files) if m and m['sf'] == sf]
    bypkt = {}
    for r in recs:
        bypkt.setdefault(r['packet'], []).append(r)
    pkts = sorted(bypkt, key=int)
    npro = max(1, int(round(len(pkts)*proto_frac)))
    proto_set = set(pkts[:npro])

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
    rng = np.random.default_rng(0)
    if len(Traw) > n_test:
        s = rng.choice(len(Traw), n_test, replace=False)
        Traw, tlab, tpkt = Traw[s], tlab[s], tpkt[s]

    print(f'SF{sf}  N={N}  M={M}   프로토타입 {npro}패킷/{len(Praw)}심볼  '
          f'평가 {len(np.unique(tpkt))}패킷/{len(Traw)}심볼')

    print('\n' + '='*76)
    print('A  올바른 생성기로 전역 오프셋 재측정')
    g_tau, g_eps = align_from_labels(Praw, plab, X)
    print(f'  해석적 생성기 기준: tau={g_tau:+.2f} 샘플, eps={g_eps:+.4f} bin')
    print(f'  (cumsum 생성기로 쟀던 값: tau=+2.00, eps=+0.188 — 그중 '
          f'tau -1.00 / eps -0.0625 는 생성기 오차였다)')
    Tal = apply_corr(Traw, g_tau, g_eps)
    Pal = apply_corr(Praw, g_tau, g_eps)

    by = {c: [] for c in range(N)}
    for y, c in zip(Pal, plab):
        if len(by[c]) < max_clean:
            by[c].append(y)
    Pemp, f0, _ = empirical_prototypes(by, N, M)
    ok = np.isfinite(f0)
    by0 = {c: [] for c in range(N)}
    for y, c in zip(Praw, plab):
        if len(by0[c]) < max_clean:
            by0[c].append(y)
    Pemp0 = empirical_prototypes(by0, N, M)[0]
    print(f'  경험 프로토타입 <-> 이상 chirp |상관|: '
          f'정렬 전 {np.abs(np.sum(Pemp0[ok]*np.conj(Xn[ok]),1)).mean():.4f}'
          f' -> 정렬 후 {np.abs(np.sum(Pemp[ok]*np.conj(Xn[ok]),1)).mean():.4f}')

    print('\n' + '='*76)
    print('B  같은 정렬을 걸고 검출기 비교')
    arms = [
        ('D1 NELoRa abs-alias (2블록)',        lambda Y: rx_alias(Y, down, osf, N, 2, 'abs')),
        ('D1 2블록 복소합',                     lambda Y: rx_alias(Y, down, osf, N, 2, 'comp')),
        ('표준 수신기 (LPF->데시->dechirp)',     lambda Y: rx_standard(Y, base_n, osf, N, True)),
        ('표준 수신기 (LPF 없이 솎기만)',         lambda Y: rx_standard(Y, base_n, osf, N, False)),
        ('D2 이상적 MF',                        lambda Y: d_mf(Y, Xn)),
        ('D3 경험적 MF',                        lambda Y: d_mf(Y, Pemp)),
    ]
    out = ser_table(arms, Tal, tlab)
    show(out)

    print('\n' + '='*76)
    print(f'C  패킷 단위 부트스트랩 CI ({nboot} 복제, 잡음 실현은 고정)')
    errs = {}
    for nm, fn in arms:
        rg = np.random.default_rng(1)
        errs[nm] = {s: (fn(add_awgn(Tal, s, rg)) != tlab) for s in SNRS}
    print(f"  {'검출기':<34}{'10% SER 중앙값':>15}{'95% CI':>22}")
    med = {}
    for nm, _ in arms:
        m, lo, hi = boot_cross(errs[nm], tpkt, SNRS, nboot)
        med[nm] = (m, lo, hi)
        print(f'  {nm:<34}{("없음" if m is None else f"{m:8.2f} dB"):>15}'
              f'{("" if m is None else f"[{lo:+7.2f}, {hi:+7.2f}]"):>22}')

    # 격차의 CI 는 복제별 차이로 직접 잡아야 한다
    print(f'\n  격차 (복제별 차이의 95% CI)')
    pairs = [('D1 NELoRa abs-alias (2블록)', '표준 수신기 (LPF->데시->dechirp)'),
             ('표준 수신기 (LPF->데시->dechirp)', 'D2 이상적 MF'),
             ('D1 NELoRa abs-alias (2블록)', 'D2 이상적 MF'),
             ('D2 이상적 MF', 'D3 경험적 MF')]
    rgp = np.random.default_rng(7)
    upk, inv = np.unique(tpkt, return_inverse=True)
    idx_by = [np.where(inv == i)[0] for i in range(len(upk))]
    picks = [rgp.integers(0, len(upk), len(upk)) for _ in range(nboot)]
    for a, b in pairs:
        ds = []
        for pick in picks:
            sel = np.concatenate([idx_by[i] for i in pick])
            ca = cross(SNRS, [errs[a][s][sel].mean()*100 for s in SNRS])
            cb = cross(SNRS, [errs[b][s][sel].mean()*100 for s in SNRS])
            if ca is not None and cb is not None:
                ds.append(ca - cb)
        if ds:
            d = np.array(ds)
            sig = '유의' if (np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0) else '구별안됨'
            print(f'  {a.split("(")[0].strip():<26} -> {b.split("(")[0].strip():<22}'
                  f'{np.median(d):+7.2f} dB  [{np.percentile(d,2.5):+6.2f}, '
                  f'{np.percentile(d,97.5):+6.2f}]  {sig}')


def cmd_sweep(root, sfs, n_test, proto_frac, max_clean, nboot):
    """SF 전체를 올바른 생성기로. D1 / 표준 수신기 / D2 세 팔만 (D3 은 SF7 에서만 성립)."""
    rows = []
    for sf in sfs:
        X, osf, M, N = chirp_bank(sf, 8)
        Xn = norm_rows(X); base_n = base_chirp(sf, 1); down = np.conj(X[0])
        files = sorted(glob.glob(os.path.join(root, str(sf), '**', '*.mat'), recursive=True))
        recs = [m for m in (parse_raw_name(f) for f in files) if m and m['sf'] == sf]
        bypkt = {}
        for r in recs:
            bypkt.setdefault(r['packet'], []).append(r)
        pkts = sorted(bypkt, key=int)
        npro = max(1, int(round(len(pkts)*proto_frac)))
        pro = set(pkts[:npro])
        nt = max(900, int(n_test*(2.0**(7 - sf))**0.5))

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
        rg = np.random.default_rng(0)
        if len(Traw) > nt:
            s2 = rg.choice(len(Traw), nt, replace=False)
            Traw, tlab, tpkt = Traw[s2], tlab[s2], tpkt[s2]
        g_tau, g_eps = align_from_labels(Praw, plab, X)
        Tal = apply_corr(Traw, g_tau, g_eps)
        grid = [v - 3*(sf - 7) for v in SNRS]
        arms = [('D1', lambda Y: rx_alias(Y, down, osf, N, 2, 'abs')),
                ('STD', lambda Y: rx_standard(Y, base_n, osf, N, True)),
                ('D2', lambda Y: d_mf(Y, Xn))]
        errs = {}
        for nm, fn in arms:
            rr = np.random.default_rng(1)
            errs[nm] = {v: (fn(add_awgn(Tal, v, rr)) != tlab) for v in grid}
        med = {nm: boot_cross(errs[nm], tpkt, grid, nboot)[0] for nm, _ in arms}
        upk, inv = np.unique(tpkt, return_inverse=True)
        idx_by = [np.where(inv == i)[0] for i in range(len(upk))]
        rgp = np.random.default_rng(7)
        ds = []
        for _ in range(nboot):
            sel = np.concatenate([idx_by[i] for i in rgp.integers(0, len(upk), len(upk))])
            a = cross(grid, [errs['D1'][v][sel].mean()*100 for v in grid])
            b = cross(grid, [errs['STD'][v][sel].mean()*100 for v in grid])
            if a is not None and b is not None:
                ds.append(a - b)
        d = np.array(ds) if ds else np.array([np.nan])
        rows.append((sf, g_tau, g_eps, med['D1'], med['STD'], med['D2'],
                     np.median(d), np.percentile(d, 2.5), np.percentile(d, 97.5),
                     len(Tal), len(upk)))
        print(f'  SF{sf} 완료  tau={g_tau:+.2f} eps={g_eps:+.4f}  '
              f'D1 {med["D1"]:.2f} / STD {med["STD"]:.2f} / D2 {med["D2"]:.2f}')

    print('\n' + '='*82)
    print('SF 전체 (올바른 생성기, 패킷 부트스트랩 중앙값)')
    print(f"  {'SF':>3}{'tau':>7}{'eps':>9}{'D1':>9}{'표준RX':>9}{'D2':>9}"
          f"{'D1->표준RX (95% CI)':>26}")
    for (sf, t, e, d1, st, d2, m, lo, hi, n, npk) in rows:
        print(f'  {sf:>3}{t:>7.2f}{e:>9.4f}{d1:>9.2f}{st:>9.2f}{d2:>9.2f}'
              f'{m:>10.2f} dB [{lo:+6.2f},{hi:+6.2f}]')
    print('\n  표준 수신기 - D2 (등가성 확인)')
    for (sf, t, e, d1, st, d2, m, lo, hi, n, npk) in rows:
        print(f'  SF{sf}: {st-d2:+.2f} dB   (평가 {npk}패킷 / {n}심볼)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['ideal', 'real', 'sweep'])
    ap.add_argument('root', nargs='?', default=None)
    ap.add_argument('--sf', type=int, default=7)
    ap.add_argument('--ntrial', type=int, default=4000)
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--proto-frac', type=float, default=0.33)
    ap.add_argument('--max-clean', type=int, default=30)
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--sfs', type=str, default='7,8,9,10')
    a = ap.parse_args()
    if a.cmd == 'ideal':
        cmd_ideal(a.sf, a.ntrial)
    elif a.cmd == 'sweep':
        cmd_sweep(a.root, [int(x) for x in a.sfs.split(',')], a.n,
                  a.proto_frac, a.max_clean, a.boot)
    else:
        cmd_real(a.root, a.sf, a.n, a.proto_frac, a.max_clean, a.boot)
