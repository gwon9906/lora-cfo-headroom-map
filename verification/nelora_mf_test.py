# -*- coding: utf-8 -*-
"""
NELoRa 데이터셋에서 검출기 세 종을 같은 심볼에 비교한다.

  D1  NELoRa baseline : dechirp -> FFT -> abs(cut1)+abs(cut2) -> argmax
  D2  이상적 MF       : 해석적으로 생성한 chirp 프로토타입과의 상관
  D3  경험적 MF       : 고SNR 캡처에서 추정한 '실제' 파형 프로토타입과의 상관

  D1->D2 격차 = 검출기 구현이 잃던 몫
  D2->D3 격차 = 이상적 chirp 가정과 실제 하드웨어 파형의 부정합 몫
  D3 이후 남는 것 = 신경망이 가져갈 수 있는 몫

사용법:
  python nelora_mf_test.py inspect <데이터폴더>      # 먼저 이걸로 구조 확인
  python nelora_mf_test.py run     <데이터폴더> [--sf 7] [--per-snr 2000]
  python nelora_mf_test.py raw     <데이터폴더> [--sf 7]   # SNR 라벨 없는 생캡처 릴리스
"""
import sys, os, re, glob, argparse
import numpy as np

FS = 1e6                      # README: sampling frequency 1 MHz


# ----------------------------------------------------------------- 로딩
def load_symbol(path):
    """.mat / .npy / .pkl 에서 복소 IQ 벡터 하나를 꺼낸다."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.mat':
        try:
            from scipy.io import loadmat
            d = loadmat(path)
        except (ValueError, TypeError):          # MAT 컨테이너가 아님 = 생 complex64
            return load_raw(path)
        except NotImplementedError:              # v7.3 = HDF5
            import h5py
            with h5py.File(path, 'r') as f:
                k = [x for x in f.keys() if not x.startswith('#')][0]
                a = f[k][()]
                if a.dtype.names and 'real' in a.dtype.names:
                    a = a['real'] + 1j*a['imag']
                return np.asarray(a).ravel().astype(np.complex128)
        for k, v in d.items():
            if not k.startswith('__') and isinstance(v, np.ndarray) and v.size > 1:
                return np.asarray(v).ravel().astype(np.complex128)
        raise ValueError(f'복소 배열을 못 찾음: {path} (키: {list(d)})')
    if ext == '.npy':
        return np.load(path).ravel().astype(np.complex128)
    if ext in ('.pkl', '.pickle'):
        import pickle
        with open(path, 'rb') as f:
            return np.asarray(pickle.load(f)).ravel().astype(np.complex128)
    raise ValueError(f'모르는 확장자: {ext}')


def load_raw(path):
    """헤더 없는 interleaved float32 IQ (NELoRa raw 캡처 릴리스)."""
    return np.fromfile(path, dtype=np.complex64).astype(np.complex128)


FIELDS = ['code_est', 'snr', 'sf', 'bw', 'batch', 'code_label', 'packet', 'symbol']

def parse_name(path):
    """{code_est}_{snr}_{sf}_{bw}_{batch}_{code_label}_{packet}_{symbol}"""
    stem = os.path.splitext(os.path.basename(path))[0]
    p = stem.split('_')
    if len(p) < 6:
        return None
    try:
        return dict(zip(FIELDS, [float(x) for x in p[:len(FIELDS)]]))
    except ValueError:
        return None


def scan(root):
    files = []
    for ext in ('*.mat', '*.npy', '*.pkl'):
        files += glob.glob(os.path.join(root, '**', ext), recursive=True)
    return sorted(files)


# ----------------------------------------------------------------- 신호
def ideal_chirps(sf, bw, fs=FS):
    """** 사용 금지 — nelora_chirp.chirp_bank 를 쓸 것. **

    위상을 cumsum 으로 적분하므로 샘플율에 따라 결과가 달라진다:
      - OSF=8 판을 데시메이션한 것과 OSF=1 판의 |상관| 이 0.71 뿐이라
        표준 수신기(데시메이션 우선)를 짜면 무잡음에서도 23% 밖에 못 맞춘다
      - X[c] != roll(X[0], -OSF*c)  (최대오차 2.0)
      - 해석적 생성기와 정확히 (tau=-1.00 샘플, eps=-0.0625 bin) 어긋나 있어,
        데이터셋 오프셋 측정값에 그만큼이 섞여 들어간다
    아래 함수들은 이 뱅크를 쓰던 초기 실행을 재현하기 위해 남겨 둔다."""
    N = 2**sf
    osf = int(round(fs/bw))
    M = osf*N
    n = np.arange(M)
    out = np.empty((N, M), dtype=np.complex128)
    for c in range(N):
        f = (((c + n/osf) % N)/N - 0.5)*bw
        out[c] = np.exp(1j*2*np.pi*np.cumsum(f)/fs)
    return out, osf, M, N


def norm_rows(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-30)


# ------------------------------------------------------------- 검출기
def d1_nelora(Y, down, osf, U=1):
    """abs(cut1)+abs(cut2). 원본 chirp_abs_alias 와 동일."""
    M = Y.shape[1]
    Z = np.fft.fft(Y*down[None, :], M*U, axis=1)
    T = int(round(M*U/osf))
    s = np.abs(Z[:, :T]) + np.abs(Z[:, -T:])
    return (np.round(np.argmax(s, 1)/U).astype(int)) % (M//osf)


def d_mf(Y, P):
    """비동기 정합필터 뱅크: argmax_m |<y, p_m>|"""
    return np.argmax(np.abs(Y @ np.conj(P).T), 1)


def empirical_prototypes(clean_by_code, N, M):
    """코드별 고SNR 캡처들의 주성분(= 임의 위상에 불변인 ML 추정)."""
    P = np.zeros((N, M), dtype=np.complex128)
    frac, cnt = np.full(N, np.nan), np.zeros(N, int)
    for c, lst in clean_by_code.items():
        if not lst:
            continue
        A = norm_rows(np.array(lst))
        cnt[c] = len(A)
        if len(A) == 1:
            P[c] = A[0]; frac[c] = 1.0; continue
        _, s, Vh = np.linalg.svd(A, full_matrices=False)
        P[c] = Vh[0]
        frac[c] = (s[0]**2)/(s**2).sum()      # 1 에 가까우면 캡처들이 서로 일관됨
    return P, frac, cnt


# ------------------------------------------------------------------ 실행
def cmd_inspect(root):
    files = scan(root)
    print(f'파일 {len(files)}개')
    if not files:
        print('  -> 확장자가 mat/npy/pkl 이 아닙니다. 폴더 안을 직접 확인하세요.'); return
    for f in files[:5]:
        print('  ', os.path.relpath(f, root))
    meta = [m for m in (parse_name(f) for f in files[:3000]) if m]
    print(f'\n파일명 파싱 성공 {len(meta)}/{min(len(files),3000)}')
    if meta:
        for k in FIELDS:
            v = sorted({m[k] for m in meta})
            print(f'  {k:11s}: {len(v):5d}종  예) {v[:8]}')
    else:
        raw = [m for m in (parse_raw_name(f) for f in files[:3000]) if m]
        print(f'raw 레이아웃 파싱 성공 {len(raw)}/{min(len(files),3000)}')
        if raw:
            for k in ('sf', 'code_label', 'symbol'):
                v = sorted({m[k] for m in raw})
                print(f'  {k:11s}: {len(v):5d}종  예) {v[:8]}')
            print(f'  packet     : {len({m["packet"] for m in raw}):5d}종')
            print('  -> SNR 라벨이 없습니다. raw 모드를 쓰세요.')
    x = load_symbol(files[0])
    print(f'\n첫 심볼: shape={x.shape} dtype={x.dtype} '
          f'|x| 평균={np.abs(x).mean():.4f} 복소={np.iscomplexobj(x)}')
    if meta:
        sf, bw = int(meta[0]['sf']), float(meta[0]['bw'])
        bw = bw*1000 if bw < 10000 else bw
        print(f'기대 길이 = OSF*2^SF = {int(FS/bw)}*{2**sf} = {int(FS/bw)*2**sf}')


def cmd_run(root, sf_want, per_snr, clean_snr, max_clean):
    files = scan(root)
    recs = []
    for f in files:
        m = parse_name(f)
        if m and int(m['sf']) == sf_want:
            m['path'] = f; recs.append(m)
    if not recs:
        print('해당 SF 파일 없음. inspect 먼저 실행하세요.'); return

    bw = float(recs[0]['bw']);  bw = bw*1000 if bw < 10000 else bw
    Pid, osf, M, N = ideal_chirps(sf_want, bw)
    down = np.conj(Pid[0])
    Pid = norm_rows(Pid)
    print(f'SF{sf_want}  BW={bw/1e3:.0f}kHz  OSF={osf}  N={N}  심볼길이={M}')

    # --- 라벨 신뢰도: code_est 와 code_label 이 고SNR에서 일치하는가
    cl = [r for r in recs if r['snr'] == clean_snr]
    if cl:
        agree = np.mean([r['code_est'] % N == r['code_label'] % N for r in cl])*100
        print(f'\n고SNR(SNR={clean_snr:g}) 에서 code_est == code_label : {agree:.2f}%  (n={len(cl)})')
        print('  -> 100%가 아니면 학습 라벨이 재래식 디코더 오류를 물려받고 있다는 뜻')

    # --- D3 프로토타입
    if not cl:
        print(f'\nSNR={clean_snr:g} 파일이 없어 D3(경험적 MF)는 건너뜁니다.'); Pemp = None
    else:
        by = {c: [] for c in range(N)}
        for r in cl:
            c = int(r['code_label']) % N
            if len(by[c]) < max_clean:
                by[c].append(load_symbol(r['path']))
        Pemp, frac, cnt = empirical_prototypes(by, N, M)
        ok = np.isfinite(frac)
        print(f'\n경험적 프로토타입: 코드 {ok.sum()}/{N}개, 코드당 평균 {cnt[ok].mean():.1f}개 캡처')
        print(f'  1st 특이값 에너지 비중 평균 {np.nanmean(frac):.3f} (1에 가까울수록 캡처가 일관)')
        if np.nanmean(frac) < 0.8:
            print('  ** 낮습니다. 캡처 간 타이밍/CFO 편차가 큽니다. D3 결과를 신중히 보세요.')
        # 이상적 chirp 와 얼마나 다른가
        sim = np.abs(np.sum(Pemp[ok]*np.conj(Pid[ok]), 1))
        print(f'  이상적 chirp 와의 |상관| 평균 {sim.mean():.4f} '
              f'(1이면 동일, 낮을수록 하드웨어 왜곡이 큼)')

    # --- SNR 스윕
    snrs = sorted({r['snr'] for r in recs if r['snr'] != clean_snr})
    rng = np.random.default_rng(0)
    print(f"\n{'SNR':>6}{'n':>7}{'D1 NELoRa':>12}{'D2 이상MF':>12}{'D3 경험MF':>12}")
    rows = []
    for snr in snrs:
        sub = [r for r in recs if r['snr'] == snr]
        if len(sub) > per_snr:
            sub = [sub[i] for i in rng.choice(len(sub), per_snr, replace=False)]
        Y, lab = [], []
        for r in sub:
            y = load_symbol(r['path'])
            if y.size != M:
                continue
            Y.append(y); lab.append(int(r['code_label']) % N)
        if not Y:
            continue
        Y = np.array(Y); lab = np.array(lab)
        e1 = (d1_nelora(Y, down, osf) != lab).mean()*100
        e2 = (d_mf(Y, Pid) != lab).mean()*100
        e3 = (d_mf(Y, Pemp) != lab).mean()*100 if Pemp is not None else np.nan
        rows.append((snr, e1, e2, e3))
        print(f'{snr:>6.0f}{len(Y):>7}{e1:>11.1f}%{e2:>11.1f}%{e3:>11.1f}%')

    report(rows)


def cross(xs, ys, t=10.0):
    for i in range(len(xs)-1):
        if (ys[i]-t)*(ys[i+1]-t) <= 0 and ys[i] != ys[i+1]:
            return xs[i] + (t-ys[i])*(xs[i+1]-xs[i])/(ys[i+1]-ys[i])
    return None


def report(rows, note_synth=False):
    rows = sorted(rows)
    xs = [r[0] for r in rows]
    c = [cross(xs, [r[k] for r in rows]) for k in (1, 2, 3)]
    tag = ' (합성 AWGN 기준)' if note_synth else ''
    print(f'\n10% SER 도달 SNR{tag}')
    for nm, v in zip(('D1 NELoRa baseline', 'D2 이상적 MF', 'D3 경험적 MF'), c):
        print(f'  {nm:22s} {"교차 없음" if v is None else f"{v:7.2f} dB"}')
    if c[0] is not None and c[1] is not None:
        print(f'\n  D1 -> D2 (검출기 구현)      = {c[0]-c[1]:+.2f} dB')
    if c[1] is not None and c[2] is not None:
        print(f'  D2 -> D3 (하드웨어 파형)    = {c[1]-c[2]:+.2f} dB')
    if c[0] is not None and c[2] is not None:
        print(f'  D1 -> D3 (신경망 없이 총합) = {c[0]-c[2]:+.2f} dB')
    print('  참고: 논문이 신경망으로 보고한 이득 = 1.84 ~ 2.35 dB')
    print('\n주의: CFO/타이밍 보상을 세 검출기 모두에 적용하지 않았습니다.')
    print('      NELoRa 의 DNN 경로는 phase search 를 거치므로, 최종 비교에는 동일 보상이 필요합니다.')


# ------------------------------------- raw 캡처 릴리스 (SNR 라벨 없음)
def parse_raw_name(path):
    """<SF>/<packet>/{symbol_idx}_{code_label}_{?}_{sf}.mat  (생 complex64)"""
    stem = os.path.splitext(os.path.basename(path))[0]
    p = stem.split('_')
    if len(p) != 4:
        return None
    try:
        si, code, _x, sf = (int(v) for v in p)
    except ValueError:
        return None
    return dict(symbol=si, code_label=code, sf=sf,
                packet=os.path.basename(os.path.dirname(path)), path=path)


def add_awgn(Y, snr_db, rng):
    """심볼별 측정 전력 기준으로 AWGN 을 더한다 (NELoRa 가 쓰는 방식)."""
    Ps = (np.abs(Y)**2).mean(1, keepdims=True)
    sig = np.sqrt(Ps/(10**(snr_db/10.0)))
    n = (rng.standard_normal(Y.shape) + 1j*rng.standard_normal(Y.shape))/np.sqrt(2)
    return Y + sig*n


def cmd_run_raw(root, sf_want, per_snr, snrs, proto_pkts, max_clean):
    files = scan(root)
    recs = [m for m in (parse_raw_name(f) for f in files) if m and m['sf'] == sf_want]
    if not recs:
        print('해당 SF 의 raw 파일 없음.'); return

    bw = 125e3                                  # 전 SF 공통: 8192B/SF7 -> OSF=8
    Pid, osf, M, N = ideal_chirps(sf_want, bw)
    down = np.conj(Pid[0])
    Pid = norm_rows(Pid)
    print(f'SF{sf_want}  BW={bw/1e3:.0f}kHz  OSF={osf}  N={N}  심볼길이={M}')
    print('주의: 이 릴리스에는 SNR 라벨이 없습니다. 아래 SNR 은 캡처에 *합성 AWGN* 을 더해 만든')
    print('      값이며, 캡처 자체가 이미 잡음을 포함하므로 실제 SNR 은 표기값보다 약간 낮습니다.')

    # 패킷 단위로 프로토타입용 / 평가용을 완전히 분리한다.
    # (같은 캡처로 프로토타입을 만들고 그 캡처를 평가하면 D3 이 자기 잡음을 알아보게 되어
    #  D2->D3 격차가 없는데도 있는 것처럼 나온다.)
    pkts = sorted({r['packet'] for r in recs}, key=lambda x: int(x))
    proto_set = set(pkts[:proto_pkts])
    proto = [r for r in recs if r['packet'] in proto_set and r['symbol'] >= 12]
    test  = [r for r in recs if r['packet'] not in proto_set and r['symbol'] >= 12]
    print(f'\n패킷 {len(pkts)}개 -> 프로토타입 {len(proto_set)}개({len(proto)}심볼) / '
          f'평가 {len(pkts)-len(proto_set)}개({len(test)}심볼), 서로 겹치지 않음')
    print('  (preamble/SFD 인 symbol_idx<12 는 제외, payload 심볼만 사용)')

    # --- D3 프로토타입: 프로토타입 패킷의 캡처들로만 구성
    by = {c: [] for c in range(N)}
    for r in proto:
        c = r['code_label'] % N
        if len(by[c]) < max_clean:
            y = load_raw(r['path'])
            if y.size == M:
                by[c].append(y)
    Pemp, frac, cnt = empirical_prototypes(by, N, M)
    ok = np.isfinite(frac)
    print(f'\n경험적 프로토타입: 코드 {ok.sum()}/{N}개, 코드당 평균 {cnt[ok].mean():.1f}개 캡처')
    print(f'  1st 특이값 에너지 비중 평균 {np.nanmean(frac):.3f} (1에 가까울수록 캡처가 일관)')
    if np.nanmean(frac) < 0.8:
        print('  ** 낮습니다. 캡처 간 타이밍/CFO 편차가 큽니다. D3 결과를 신중히 보세요.')
    sim = np.abs(np.sum(Pemp[ok]*np.conj(Pid[ok]), 1))
    print(f'  이상적 chirp 와의 |상관| 평균 {sim.mean():.4f} '
          f'(1이면 동일, 낮을수록 하드웨어 왜곡이 큼)')
    if ok.sum() < N:
        print(f'  ** 캡처가 없는 코드 {N-ok.sum()}개는 프로토타입이 0 벡터입니다.')

    # --- 평가 심볼 적재 (한 번만 읽고 SNR 마다 재사용)
    rng = np.random.default_rng(0)
    if len(test) > per_snr:
        test = [test[i] for i in rng.choice(len(test), per_snr, replace=False)]
    Y, lab = [], []
    for r in test:
        y = load_raw(r['path'])
        if y.size == M:
            Y.append(y); lab.append(r['code_label'] % N)
    Y = np.array(Y); lab = np.array(lab)
    print(f'\n평가 심볼 {len(Y)}개 적재')

    print(f"\n{'SNR':>6}{'n':>7}{'D1 NELoRa':>12}{'D2 이상MF':>12}{'D3 경험MF':>12}")
    rows = []
    for snr in [None] + list(snrs):
        Yn = Y if snr is None else add_awgn(Y, snr, rng)
        e1 = (d1_nelora(Yn, down, osf) != lab).mean()*100
        e2 = (d_mf(Yn, Pid) != lab).mean()*100
        e3 = (d_mf(Yn, Pemp) != lab).mean()*100
        tag = '원본' if snr is None else f'{snr:.0f}'
        print(f'{tag:>6}{len(Yn):>7}{e1:>11.1f}%{e2:>11.1f}%{e3:>11.1f}%')
        if snr is not None:
            rows.append((snr, e1, e2, e3))

    report(rows, note_synth=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['inspect', 'run', 'raw'])
    ap.add_argument('root')
    ap.add_argument('--sf', type=int, default=7)
    ap.add_argument('--per-snr', type=int, default=2000)
    ap.add_argument('--clean-snr', type=float, default=35)
    ap.add_argument('--max-clean', type=int, default=30)
    ap.add_argument('--snrs', type=str, default='0,-5,-10,-12,-14,-16,-18,-20',
                    help='raw 모드에서 합성할 AWGN SNR 목록 (dB)')
    ap.add_argument('--proto-pkts', type=int, default=60,
                    help='raw 모드에서 D3 프로토타입 전용으로 떼어둘 패킷 수')
    a = ap.parse_args()
    a.snrs = [float(x) for x in a.snrs.split(',')]
    if a.cmd == 'inspect':
        cmd_inspect(a.root)
    elif a.cmd == 'raw':
        cmd_run_raw(a.root, a.sf, a.per_snr, a.snrs, a.proto_pkts, a.max_clean)
    else:
        cmd_run(a.root, a.sf, a.per_snr, a.clean_snr, a.max_clean)
