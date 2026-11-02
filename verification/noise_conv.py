# -*- coding: utf-8 -*-
"""잡음 규약이 SER 곡선을 어디로 옮기는가 — 이론과의 3 dB 를 쪼갠다.

**질문.** `theory_ser.py` 는 SF7 / 10% SER 의 이론 교차점을 전대역 축에서
−20.02 dB (in-band −10.99) 로 준다. 그런데 실캡처 held-out 에서 가장 좋은 팔
(D3 정렬된 경험적 MF)이 −17.05 dB 이고, 곡선의 기울기도 이론보다 완만하다.
3 dB 가 어디서 오는가?

**가설** (지도 지적). 그들 `add_noise` 는 잡음 진폭을 **배치 평균**으로 정한다:

    amp = 10^(-snr/20) * torch.mean(torch.abs(dataY))   # 배치 스칼라
    noise = amp/sqrt(2) * (randn(num_samples) + 1j*randn(num_samples))  # (M,) 하나

두 가지가 겹쳐 있다 — (i) **진폭이 배치 스칼라**라 캡처 진폭이 배치 평균과 다른
심볼은 표기 SNR 과 다른 실제 SNR 을 받는다. SER 곡선은 볼록하므로 SNR 이 흩어지면
곡선이 평균되어 완만해지고 10% 교차점이 **오른쪽으로** 밀린다 (Jensen).
(ii) **잡음 실현 하나가 배치 전체에 브로드캐스트**된다 — 심볼간 독립성이 깨진다.

**설계.** (i) 과 (ii) 를 분리해서 네 규약을 같은 심볼·같은 SNR 격자에서 잰다:

  theirs          배치 스칼라 amp + 배치 공유 잡음        <- 그들 코드 그대로
  batch-amp-iid   배치 스칼라 amp + 심볼별 독립 잡음      <- (ii) 만 제거
  sliding         슬라이딩 윈도우 amp + 독립 잡음          <- 실수신기 AGC 에 가깝다
  per-symbol      심볼별 amp + 독립 잡음                  <- (i)(ii) 모두 제거

그리고 두 모집단에서 잰다:

  A. **이상 처프** — 캡처 잡음도, 타이밍/CFO 잔차도 없다. 규약 효과만 남는다.
     여기서 per-symbol 이 이론과 붙으면 이론식과 하네스가 둘 다 맞다는 뜻이다.
  B. **실캡처 held-out** — 규약 + 캡처 잡음 + 잔여 임페어먼트 전부.

A 와 B 의 차이가 "캡처가 이미 잡음을 포함한다" 는 유보(NELORA.md 머리말)의 크기다.

**중요 — 이 스크립트는 헤드라인을 바꾸지 않는다.** 헤드라인은 팔 사이의 **격차**이고,
규약은 모든 팔에 똑같이 걸린다. 여기서 확인할 것은 (a) 절대 dB 축이 규약에 얼마나
민감한가, (b) 격차가 규약에 불변인가 — 둘째가 깨지면 헤드라인이 흔들린다.

사용법:
  python noise_conv.py ideal
  python noise_conv.py real --cache results/cache_sf7.npz \\
      --only-idx results/split_sf7_test.npy --out results/noise_conv_real.json
"""
import argparse
import json
import os

import numpy as np

from nelora_dnn_eval import (add_noise_theirs, align_from_labels_np, apply_corr_np,
                             base_chirp_n, build_params, chirp_bank_np, cross_np,
                             d_mf_np, decode_loraphy_batch, empirical_prototypes_np,
                             provenance, standard_rx)
from loraphy_orig import decode_loraphy_orig
from theory_ser import Conv, esn0_at_ser, ser_noncoherent

MODES = ('theirs', 'batch-amp-iid', 'sliding', 'per-symbol')


# ------------------------------------------------------------------ 잡음
def amp_ref(Y, mode, batch=16, win=8):
    """심볼마다 **잡음 진폭의 기준**이 되는 mean|y| 를 돌려준다, (n,) 배열.

    네 규약의 차이는 전부 여기에 있다 — 그들 규약(`amp = 10^(-snr/20)·mean|y|`)은
    그대로 두고, **그 평균을 어느 범위에서 잡는가** 만 바꾼다.
    (`theirs` 는 여기에 더해 잡음 실현까지 배치가 공유한다 — add_noise 참조.)

    ** 전체 스트림에 대해 한 번 계산한다. ** 배치 슬라이스마다 따로 계산하면
    sliding 윈도가 배치 경계에서 잘려 실수신기 AGC 와 달라지고, amp_spread 진단과도
    어긋난다. 그래서 측정 루프는 이 배열을 미리 받아 슬라이스만 해서 쓴다.
    """
    a = np.abs(Y).mean(1).astype(np.float64)
    if mode == 'per-symbol':
        return a
    if mode in ('theirs', 'batch-amp-iid'):
        ref = np.empty_like(a)
        for i in range(0, len(a), batch):
            ref[i:i+batch] = a[i:i+batch].mean()
        return ref
    if mode == 'sliding':
        # 길이 win 의 중심 슬라이딩 평균. 'same' 에 끝단 보정을 둬서 첫/마지막
        # 심볼이 0 으로 희석되지 않게 한다 (실수신기 AGC 와 같은 취지).
        k = np.ones(win)
        return np.convolve(a, k, 'same')/np.convolve(np.ones_like(a), k, 'same')
    raise ValueError(mode)


def add_noise(Y, snr_db, rng, mode, ref, normalize=True):
    """`ref` 는 amp_ref 가 준 이 슬라이스의 기준 진폭 (len(Y),).

    `theirs` / `per-symbol` 은 add_noise_theirs 와 **같은 연산**이어야 한다
    (selftest 가 그것을 확인한다).
    """
    g = 10.0**(-snr_db/20.0)
    amp = (g*np.asarray(ref, float))[:, None]
    if mode == 'theirs':
        # 잡음 실현 하나를 배치 전체에 브로드캐스트한다 (그들 코드 그대로).
        w = (rng.standard_normal(Y.shape[1])
             + 1j*rng.standard_normal(Y.shape[1]))[None, :]
    else:
        w = rng.standard_normal(Y.shape) + 1j*rng.standard_normal(Y.shape)
    X = Y + (amp/np.sqrt(2))*w
    if normalize:
        # 그들 코드의 정규화도 배치 스칼라다. argmax 검출기는 스케일 불변이라
        # 어느 쪽이든 결과가 같지만, 규약을 섞지 않기 위해 같이 맞춘다.
        X = X/(np.abs(X).mean() if mode == 'theirs'
               else np.abs(X).mean(1, keepdims=True))
    return X.astype(np.complex64)


def amp_spread(Y, batch=16, win=8):
    """표기 SNR 과 실제 SNR 의 차 (dB). 규약별로 심볼마다 얼마나 흩어지는가.

    실제 SNR − 표기 SNR = 20log10( mean|y_i| / (그 규약의 기준 진폭) ) 이다.
    per-symbol 은 정의상 0 이고, 나머지는 캡처 진폭의 분산만큼 흩어진다.
    """
    a = np.abs(Y).mean(1).astype(np.float64)
    return {m: 20*np.log10(a/amp_ref(Y, m, batch, win)) for m in MODES}


# ------------------------------------------------------------------ 측정
def measure(X, y, arms, snrs, modes, reals, batch, win=8, seed=1000):
    """ok[mode][arm] = (R, len(snrs), n) bool. 규약만 바꿔 같은 심볼을 돌린다."""
    n = len(X)
    ok = {m: {k: np.zeros((reals, len(snrs), n), bool) for k in arms} for m in modes}
    for m in modes:
        ref = amp_ref(X, m, batch, win)        # 스트림 전체에 대해 한 번
        for r in range(reals):
            rng = np.random.default_rng(seed + r)
            for j, s in enumerate(snrs):
                for i in range(0, n, batch):
                    Yb, lb = X[i:i+batch], y[i:i+batch]
                    Yn = add_noise(Yb, s, rng, m, ref[i:i+batch])
                    for k, f in arms.items():
                        ok[m][k][r, j, i:i+len(lb)] = (f(Yn) == lb)
            print(f'    {m:<14} 실현 {r+1}/{reals}', flush=True)
    return ok


def crossings(ok, snrs, thr=10.0):
    """팔별 10% 교차점 (R 실현 평균 곡선에서)."""
    out = {}
    for k, arr in ok.items():
        ser = [(1 - arr[:, j, :].mean())*100 for j in range(len(snrs))]
        out[k] = (cross_np(list(snrs), ser, thr), ser)
    return out


def slope(snrs, ser, lo=10.0, hi=90.0):
    """SER hi% -> lo% 를 몇 dB 에 지나는가. 이론은 SF7 에서 8.56 dB."""
    a, b = cross_np(list(snrs), ser, hi), cross_np(list(snrs), ser, lo)
    return None if (a is None or b is None) else b - a


def boot_gap(ok, snrs, pk, pairs, nboot=1000, seed=7):
    """패킷 재표집으로 격차 CI. ok: {arm: (R, S, n)}"""
    if pk is None:
        return {}
    rgb = np.random.default_rng(seed)
    upk, inv = np.unique(pk, return_inverse=True)
    idx_by = [np.where(inv == i)[0] for i in range(len(upk))]
    R = next(iter(ok.values())).shape[0]
    acc = {p: [] for p in pairs}
    for _ in range(nboot):
        r = int(rgb.integers(R))
        sel = np.concatenate([idx_by[i] for i in rgb.integers(0, len(upk), len(upk))])
        cr = {k: cross_np(list(snrs),
                          [(1 - a[r, j, sel].mean())*100 for j in range(len(snrs))])
              for k, a in ok.items()}
        for p in pairs:
            if cr.get(p[0]) is not None and cr.get(p[1]) is not None:
                acc[p].append(cr[p[0]] - cr[p[1]])
    out = {}
    for p, v in acc.items():
        if v:
            v = np.array(v)
            out[f'{p[0]}_minus_{p[1]}'] = dict(
                median=float(np.median(v)), lo=float(np.percentile(v, 2.5)),
                hi=float(np.percentile(v, 97.5)), n=len(v))
    return out


# ------------------------------------------------------------------ 온전성 검사
def selftest(sf=7):
    """규약 구현 검사. 통과하지 못하면 아래 숫자를 읽을 이유가 없다."""
    print('='*78)
    print('T0  규약 구현 온전성')
    P = build_params(sf)
    rng0 = np.random.default_rng(5)
    X = chirp_bank_np(sf, 8)[rng0.integers(0, P['N'], 32)]
    X = (X*np.exp(2j*np.pi*rng0.random((32, 1)))).astype(np.complex64)
    ok = True

    # (a) theirs / per-symbol 은 add_noise_theirs 와 같은 연산이어야 한다.
    #     (배치 하나를 그대로 넘기므로 amp_ref 의 범위가 그들 코드와 일치한다)
    #
    #     허용오차는 float32 반올림 수준이다. 그들 코드는 amp 를 complex64 입력에서
    #     float32 로 잡고, amp_ref 는 같은 float32 평균을 float64 로 넓혀 쓴다.
    #     따라서 amp/sqrt(2) 의 반올림만 ~1e-7 상대로 다르고 그것이 전부다.
    #     0 을 요구하면 정밀도를 일부러 낮춰야 하므로 그렇게 하지 않는다.
    TOL = 1e-6
    for m in ('theirs', 'per-symbol'):
        A = add_noise(X, -15.0, np.random.default_rng(1), m,
                      amp_ref(X, m, len(X)))
        B = add_noise_theirs(X, -15.0, np.random.default_rng(1), m, True, None)
        d = float(np.abs(A - B).max()/np.abs(B).max())
        print(f'  {m:<14} add_noise_theirs 와 상대차 {d:.2e} (허용 {TOL:.0e})  '
              + ('OK' if d < TOL else '** 불일치'))
        ok &= d < TOL

    # (b) 진폭이 모두 같으면 네 규약의 amp 가 같아야 한다 (이상 처프는 정포락선)
    Xc = X/np.abs(X).mean(1, keepdims=True)     # 심볼별 mean|y| = 1 로 맞춤
    sp = amp_spread(Xc)
    print('  정포락선·동일진폭 입력에서 규약별 SNR 흩어짐 (dB, 최대 |값|):')
    for m in MODES:
        v = float(np.abs(sp[m]).max())
        print(f'    {m:<14}{v:8.2e}  ' + ('OK' if v < 1e-5 else '** 0 이 아니다'))
        ok &= v < 1e-5

    # (c) 잡음 전력이 규약대로인가 — 전대역 SNR 정의 확인
    s = -10.0
    A = add_noise(Xc, s, np.random.default_rng(2), 'per-symbol',
                  amp_ref(Xc, 'per-symbol'), normalize=False)
    nz = A - Xc
    got = 10*np.log10((np.abs(Xc)**2).mean()/(np.abs(nz)**2).mean())
    print(f'  per-symbol, 표기 {s:+.1f} dB -> 실측 샘플당 SNR {got:+.3f} dB  '
          + ('OK' if abs(got - s) < 0.1 else '** 어긋남'))
    ok &= abs(got - s) < 0.1

    print('\n  ' + ('OK' if ok else '** 실패: 여기서 멈출 것.'))
    return ok


# ------------------------------------------------------------------ A. 이상 처프
def cmd_ideal(a):
    P = build_params(a.sf)
    N, osf = P['N'], P['osf']
    bank = chirp_bank_np(a.sf, osf)
    bank_n = bank/np.linalg.norm(bank, axis=1, keepdims=True)
    base_n = base_chirp_n(a.sf)
    rng = np.random.default_rng(0)
    y = rng.integers(0, N, a.ntrial)
    X = (bank[y]*np.exp(2j*np.pi*rng.random((a.ntrial, 1)))).astype(np.complex64)

    arms = {
        'decode_loraphy': lambda Y: decode_loraphy_batch(Y, P, a.upsampling),
        'loraphy_orig':   lambda Y: decode_loraphy_orig(Y, a.sf, osf, 10, 'brick'),
        'standard_rx':    lambda Y: standard_rx(Y, P, base_n),
        'D2_aligned':     lambda Y: d_mf_np(Y.astype(np.complex128), bank_n),
    }
    snrs = [float(v) for v in a.snrs.split(',')]
    print('\n' + '='*78)
    print(f'A  이상 처프 (SF{a.sf}, {a.ntrial}심볼, 캡처잡음·임페어먼트 없음)')
    print('   규약 효과만 남는다. per-symbol 이 이론과 붙어야 한다.')
    ok = measure(X, y, arms, snrs, a.modes, a.reals, a.batch, a.win)
    return report(ok, snrs, list(arms), a, None, 'ideal')


# ------------------------------------------------------------------ B. 실캡처
def cmd_real(a):
    P = build_params(a.sf)
    N, osf = P['N'], P['osf']
    d = np.load(a.cache, allow_pickle=True)
    X_all, y_all, pk_all = d['X'], d['y'], d['pk']
    print(f'캐시 {a.cache}: {len(X_all)}심볼')
    idx = np.load(a.only_idx)
    held = np.zeros(len(X_all), bool)
    held[idx] = True
    X, y, pk = X_all[idx], y_all[idx], pk_all[idx]
    print(f'held-out {len(X)}심볼 / {len(np.unique(pk))}패킷')

    bank = chirp_bank_np(a.sf, osf)
    bank_n = bank/np.linalg.norm(bank, axis=1, keepdims=True)
    base_n = base_chirp_n(a.sf)

    # D3 프로토타입은 held-out 밖에서만 (기존 측정과 같은 절차)
    Xp, yp = X_all[~held].astype(np.complex128), y_all[~held]
    g_tau, g_eps = align_from_labels_np(Xp, yp, bank)
    print(f'기준선 정렬: tau={g_tau:+.2f} 샘플, eps={g_eps:+.4f} bin')
    Pal = apply_corr_np(Xp, g_tau, g_eps)
    by = {c: [] for c in range(N)}
    for v, c in zip(Pal, yp):
        if len(by[c]) < 30:
            by[c].append(v)
    Pemp = empirical_prototypes_np(by, N, P['M'])

    arms = {
        'decode_loraphy': lambda Y: decode_loraphy_batch(Y, P, a.upsampling),
        'loraphy_orig':   lambda Y: decode_loraphy_orig(Y, a.sf, osf, 10, 'brick'),
        'standard_rx':    lambda Y: standard_rx(Y, P, base_n),
        'D2_aligned':     lambda Y: d_mf_np(apply_corr_np(Y.astype(np.complex128),
                                                          g_tau, g_eps), bank_n),
        'D3_aligned':     lambda Y: d_mf_np(apply_corr_np(Y.astype(np.complex128),
                                                          g_tau, g_eps), Pemp),
    }
    snrs = [float(v) for v in a.snrs.split(',')]

    print('\n' + '='*78)
    print('B0  캡처 진폭이 실제로 얼마나 흩어지는가 (표기 SNR 과의 차, dB)')
    sp = amp_spread(X, a.batch, a.win)
    print(f"  {'규약':<16}{'std':>8}{'5%':>8}{'95%':>8}{'최소':>8}{'최대':>8}")
    spread = {}
    for m in MODES:
        v = sp[m]
        spread[m] = dict(std=float(v.std()), p5=float(np.percentile(v, 5)),
                         p95=float(np.percentile(v, 95)),
                         min=float(v.min()), max=float(v.max()))
        print(f'  {m:<16}{v.std():8.2f}{np.percentile(v,5):8.2f}'
              f'{np.percentile(v,95):8.2f}{v.min():8.2f}{v.max():8.2f}')

    print('\n' + '='*78)
    print(f'B  실캡처 held-out (SF{a.sf})')
    ok = measure(X, y, arms, snrs, a.modes, a.reals, a.batch, a.win)
    return report(ok, snrs, list(arms), a, pk, 'real', spread)


# ------------------------------------------------------------------ 보고
def report(ok, snrs, arms, a, pk, kind, spread=None):
    c = Conv(a.sf, 8)
    th10 = c.esn0_to_fullband(esn0_at_ser(0.10, c.N))
    th_slope = (esn0_at_ser(0.10, c.N) - esn0_at_ser(0.90, c.N))

    print('\n' + '='*78)
    print(f'10% SER 교차점 (전대역 축) — 이론 {th10:+.2f} dB')
    print(f"  {'팔':<18}" + ''.join(f'{m:>16}' for m in a.modes))
    out = {}
    for m in a.modes:
        out[m] = {k: dict(cross10=v[0], ser=v[1]) for k, v in
                  crossings(ok[m], snrs).items()}
    for k in arms:
        cells = []
        for m in a.modes:
            v = out[m][k]['cross10']
            cells.append(f'{v:+.2f} dB' if v is not None else '없음')
        print(f'  {k:<18}' + ''.join(f'{v:>16}' for v in cells))

    print('\n이론과의 차 (양수 = 이론보다 그만큼 더 높은 SNR 이 필요하다 = 나쁘다)')
    for k in arms:
        row = []
        for m in a.modes:
            v = out[m][k]['cross10']
            row.append(f'{v - th10:+.2f} dB' if v is not None else '없음')
        print(f'  {k:<18}' + ''.join(f'{v:>16}' for v in row))

    print(f'\n곡선의 급경사: SER 90% -> 10% 폭 (dB) — 이론 {th_slope:.2f} dB')
    for k in arms:
        row = []
        for m in a.modes:
            s = slope(snrs, out[m][k]['ser'])
            out[m][k]['slope_90_to_10'] = s
            row.append(f'{s:.2f} dB' if s is not None else '없음')
        print(f'  {k:<18}' + ''.join(f'{v:>16}' for v in row))

    pairs = [('decode_loraphy', 'standard_rx'), ('decode_loraphy', 'loraphy_orig'),
             ('loraphy_orig', 'standard_rx'), ('decode_loraphy', 'D3_aligned')]
    pairs = [p for p in pairs if p[0] in arms and p[1] in arms]
    boots = {}
    if pk is not None and a.boot:
        print(f'\n격차가 규약에 불변인가 (패킷 재표집 {a.boot} 복제)')
        print(f"  {'격차':<36}" + ''.join(f'{m:>22}' for m in a.modes))
        for m in a.modes:
            boots[m] = boot_gap(ok[m], snrs, pk, pairs, a.boot)
        for p in pairs:
            key = f'{p[0]}_minus_{p[1]}'
            cells = []
            for m in a.modes:
                b = boots[m].get(key)
                cells.append(f"{b['median']:+.2f} [{b['lo']:+.2f},{b['hi']:+.2f}]"
                             if b else '없음')
            print(f'  {key:<36}' + ''.join(f'{v:>22}' for v in cells))

    rec = dict(kind=kind, sf=a.sf, osf=8, modes=list(a.modes), snrs=snrs,
               arms=arms, upsampling=a.upsampling, batch=a.batch,
               noise_reals=a.reals, win=a.win, n_symbols=int(next(iter(
                   ok[a.modes[0]].values())).shape[2]),
               theory=dict(cross10_fullband_db=th10,
                           cross10_inband_db=th10 + c.inband_offset_db,
                           slope_90_to_10_db=th_slope,
                           inband_offset_db=c.inband_offset_db),
               results={m: {k: dict(cross10=out[m][k]['cross10'],
                                    slope_90_to_10=out[m][k]['slope_90_to_10'],
                                    ser=out[m][k]['ser']) for k in arms}
                        for m in a.modes},
               bootstrap=boots or None, amp_spread=spread,
               provenance=provenance(('noise_conv.py', 'nelora_dnn_eval.py',
                                      'loraphy_orig.py', 'theory_ser.py')))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(rec, f, indent=2)
        print(f'\n저장: {a.out}')
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['ideal', 'real', 'selftest'])
    ap.add_argument('--sf', type=int, default=7)
    ap.add_argument('--cache', default='results/cache_sf7.npz')
    ap.add_argument('--only-idx', default='results/split_sf7_test.npy')
    ap.add_argument('--out', default=None)
    ap.add_argument('--ntrial', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--win', type=int, default=8,
                    help='sliding 규약의 윈도 길이 (심볼 수)')
    ap.add_argument('--upsampling', type=int, default=100)
    ap.add_argument('--reals', type=int, default=4)
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--modes', default=','.join(MODES))
    ap.add_argument('--snrs', default=','.join(
        str(v) for v in range(-10, -31, -1)))
    ap.add_argument('--no-selftest', action='store_true')
    a = ap.parse_args()
    a.modes = tuple(a.modes.split(','))
    if a.cmd == 'selftest':
        raise SystemExit(0 if selftest(a.sf) else 1)
    if not a.no_selftest and not selftest(a.sf):
        raise SystemExit(1)
    (cmd_ideal if a.cmd == 'ideal' else cmd_real)(a)
