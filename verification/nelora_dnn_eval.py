# -*- coding: utf-8 -*-
"""
NELoRa DNN 과 표준 수신기를 '같은 배치' 에서 잰다.

이 파일은 NELoRa-Bench 저장소(daibiaoxuwu/NeLoRa_Dataset) 안에 두고 돌린다.
model_components.py 와 같은 폴더여야 한다.

설계 원칙: 그들 코드를 다시 쓰지 않는다.
  데이터 로딩·필터·잡음·SNR 축을 그대로 쓰고, 배치 루프 안에 팔 하나만 더한다.
  별도 스크립트로 표준 RX 를 돌리면 축이 미세하게 갈리고 그 차이가 3 dB 안에 섞인다.

재는 것 (전부 같은 dataX 텐서에 대해):
  A. decode_loraphy (U=100)        그들 baseline, 기준점
  B. NELoRa DNN                     maskCNN + C_XtoY
  C. 표준 수신기                     LPF(+-BW/2) -> N 샘플 데시메이션 -> de-chirp -> N-FFT
                                    학습 없음, 정렬 없음

주의 — held-out 은 불가능하다:
  train() 의 random_split 은 torch 시드를 잡지 않으므로(모듈 상단 np/random 시드는
  torch 에 영향 없음) 체크포인트가 어느 10% 를 안 봤는지 복원할 수 없다.
  따라서 이것은 as-run 측정이고, DNN 은 평가 심볼의 ~90% 를 학습에서 봤다.
  즉 이 조건은 DNN 에 유리하다. 그래도 표준 RX 가 이긴다면 그것이 결론이다.

사용법 (WSL):
  # 1) 저장소와 데이터/체크포인트 준비
  git clone --depth 1 https://github.com/daibiaoxuwu/NeLoRa_Dataset.git
  cd NeLoRa_Dataset
  cp /path/to/nelora_dnn_eval.py .

  # 2) 먼저 재현 확인 (DNN 없이 baseline 만, 빠름)
  python nelora_dnn_eval.py --sf 7 --data-dir /path/to/NeLoRa_Dataset/7/ --no-dnn

  # 3) 본 측정
  python nelora_dnn_eval.py --sf 7 --data-dir /path/to/NeLoRa_Dataset/7/ \
      --ckpt-dir checkpoint/sf7 --out result_sf7.json

  # 결과 result_sf7.json 만 가져오면 된다.
"""
import os, json, math, argparse, random
import numpy as np
# torch 는 DNN 팔에서만 필요하다. --no-dnn 이면 없어도 돌아간다.
from scipy.signal import chirp
from scipy.fft import fft as _sfft


def fft(*args, **kw):
    kw.setdefault('workers', -1)
    return _sfft(*args, **kw)


# ------------------------------------------------------------------ 설정
def build_params(sf, bw=125e3, fs=1e6):
    num_classes = 2**sf
    num_samples = int(num_classes*fs/bw)
    t = np.linspace(0, num_samples/fs, num_samples + 1)[:-1]
    # 그들 main.py 와 동일한 downchirp
    ci = chirp(t, f0=bw/2, f1=-bw/2, t1=2**sf/bw, method='linear', phi=90)
    cq = chirp(t, f0=bw/2, f1=-bw/2, t1=2**sf/bw, method='linear', phi=0)
    downchirp = ci + 1j*cq
    return dict(sf=sf, bw=bw, fs=fs, N=num_classes, M=num_samples,
                osf=int(round(fs/bw)), downchirp=downchirp)


# -------------------------------------------------- A. 그들 baseline
def decode_loraphy_scalar(data_in, num_classes, downchirp, upsampling=100):
    """그들 main.py 원문 그대로. 벡터화 판의 정답 대조용."""
    chirp_data = data_in*downchirp
    fft_raw = fft(chirp_data, len(chirp_data)*upsampling)
    target_nfft = num_classes*upsampling
    cut1 = np.array(fft_raw[:target_nfft])
    cut2 = np.array(fft_raw[-target_nfft:])
    return round(np.argmax(abs(cut1) + abs(cut2))/upsampling) % num_classes


def decode_loraphy_batch(Y, P, upsampling=100, chunk=32):
    """위와 동일한 로직을 배치로. 값이 같은지는 --selftest 에서 확인한다."""
    N, M = P['N'], P['M']
    down = P['downchirp']
    out = np.empty(len(Y), dtype=np.int64)
    T = N*upsampling
    for i in range(0, len(Y), chunk):
        Z = fft(Y[i:i+chunk]*down[None, :], M*upsampling, axis=1)  # workers=-1
        s = np.abs(Z[:, :T]) + np.abs(Z[:, -T:])
        out[i:i+chunk] = np.round(np.argmax(s, 1)/upsampling).astype(np.int64) % N
    return out


# -------------------------------------------------- C. 표준 수신기
def base_chirp_n(sf):
    """해석적 2차 위상, OSF=1 (N 샘플). 그들 downchirp 과 상수 위상만 다르다."""
    N = 2**sf
    n = np.arange(N)
    return np.exp(2j*np.pi*(n**2/(2.0*N) - n/2.0))


# ---- 판정 기준선(D2/D3)용 최소 구현. 이 파일만 복사하면 되도록 인라인한다. ----
def chirp_bank_np(sf, osf=8):
    """해석적 2차 위상 chirp 뱅크 (nelora_chirp.chirp_bank 와 동일)."""
    N = 2**sf
    M = osf*N
    n = np.arange(M)
    base = np.exp(2j*np.pi*(n**2/(2.0*N*osf**2) - n/(2.0*osf)))
    idx = (np.arange(M)[None, :] + osf*np.arange(N)[:, None]) % M
    return base[idx]


def _norm_rows(A):
    return A/(np.linalg.norm(A, axis=1, keepdims=True) + 1e-30)


def _frac_shift(X, tau):
    M = X.shape[-1]
    f = np.fft.fftfreq(M)
    return np.fft.ifft(np.fft.fft(X, axis=-1)*np.exp(-2j*np.pi*f*tau), axis=-1)


def apply_corr_np(Y, tau, eps):
    """추정한 (tau, eps) 를 되돌려 신호를 뱅크에 정렬한다."""
    M = Y.shape[-1]
    n = np.arange(M)
    return _frac_shift(Y, -tau)*np.exp(-2j*np.pi*eps*n/M)


def align_from_labels_np(Y, lab, bank, n_use=400, seed=0):
    """라벨을 아는 심볼들로 전역 (tau, eps) 추정."""
    M = bank.shape[1]
    rng = np.random.default_rng(seed)
    if len(Y) > n_use:
        s = rng.choice(len(Y), n_use, replace=False)
        Y, lab = Y[s], lab[s]
    n = np.arange(M)
    eps_grid = np.arange(-1, 1.001, 1/32)
    E = np.exp(-2j*np.pi*eps_grid[None, :]*n[:, None]/M)
    best, bt, be = -1.0, 0.0, 0.0
    for tau in np.arange(-8, 8.01, 0.25):
        Ps = _norm_rows(_frac_shift(bank, tau))
        sc = np.abs((Y*np.conj(Ps[lab])) @ E).sum(0)
        j = int(np.argmax(sc))
        if sc[j] > best:
            best, bt, be = sc[j], tau, float(eps_grid[j])
    return bt, be


def empirical_prototypes_np(by, N, M):
    """코드별 캡처들의 주성분 (임의 위상에 불변인 ML 추정)."""
    Pr = np.zeros((N, M), dtype=np.complex128)
    for c, lst in by.items():
        if not lst:
            continue
        A = _norm_rows(np.array(lst))
        if len(A) == 1:
            Pr[c] = A[0]
            continue
        Pr[c] = np.linalg.svd(A, full_matrices=False)[2][0]
    return Pr


def d_mf_np(Y, bank):
    return np.argmax(np.abs(Y @ np.conj(bank).T), 1)


def cross_np(xs, ys, t=10.0):
    """SER 곡선이 임계 t 를 지나는 SNR (선형보간).

    ** 마지막 교차를 쓴다. ** 학습 SNR 범위 밖(예: +10 dB)에서 신경망이 무너지면
    곡선이 단조롭지 않아 앞쪽에 가짜 교차가 생긴다. 실제로 LPF 팔이 +10 dB 에서
    4% 바닥을 보였고(학습은 -30~0 dB), 첫 교차를 쓰면 부트스트랩 꼬리가 -9 dB 까지
    늘어났다. 임계 부근(저SNR 쪽)은 단조로우므로 마지막 교차가 옳다.
    """
    hit = None
    for i in range(len(xs) - 1):
        if (ys[i] - t)*(ys[i+1] - t) <= 0 and ys[i] != ys[i+1]:
            hit = xs[i] + (t - ys[i])*(xs[i+1] - xs[i])/(ys[i+1] - ys[i])
    return hit


def brickwall_lpf_np(Y, P):
    """±BW/2 브릭월 LPF. OSF 와 텐서 모양은 유지 (nelora_lpf_patch 와 동일 연산)."""
    N, M = P['N'], P['M']
    Z = np.fft.fft(Y, axis=1)
    mask = np.zeros(M, dtype=bool)
    mask[:N//2] = True
    mask[-(N - N//2):] = True
    return np.fft.ifft(Z*mask, axis=1).astype(np.complex64)


def standard_rx(Y, P, base_n):
    """LPF(+-BW/2) -> N 샘플 데시메이션 -> de-chirp -> N-FFT.

    de-chirp *이전에* 거르는 것이 핵심. 대역 밖 잡음 (OSF-1)/OSF 가 버려지고
    두 alias 조각이 같은 빈에 코히런트하게 접힌다."""
    N = P['N']
    Z = np.fft.fft(Y, axis=1)
    Zd = np.concatenate([Z[:, :N//2], Z[:, -(N - N//2):]], axis=1)
    yd = np.fft.ifft(Zd, axis=1)
    return np.argmax(np.abs(np.fft.fft(yd*np.conj(base_n)[None, :], N, axis=1)), 1) % N


# -------------------------------------------------- 데이터 (그들 방식)
def load_data(P, data_dir, cache, max_symbols=None, upsampling=100, verbose=True):
    """그들 load_data(): decode_loraphy 가 무잡음에서 맞히는 심볼만 남긴다."""
    if cache and os.path.exists(cache):
        with open(cache, 'rb') as g:
            d = np.load(g, allow_pickle=True)
            return d['X'], d['y'], d['stat'].item(), d['pk']

    N, M = P['N'], P['M']
    # ** 그들 load_data 와 같은 순서여야 한다. **
    #   그들은 files[truth_idx] 에 모아두고 truth_idx 오름차순으로 돈다 (코드 순).
    #   패킷 순으로 돌면 --only-idx 의 인덱스가 다른 심볼을 가리킨다.
    buckets = [[] for _ in range(N)]
    for sub in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            buckets[int(fn.split('_')[1]) % N].append((os.path.join(d, fn), sub))
    files = [(fp, sub) for lst in buckets for (fp, sub) in lst]
    if verbose:
        print(f'  파일 {len(files)}개 발견 (코드 순 — 그들 load_data 와 동일)')

    raw = np.empty((len(files), M), dtype=np.complex64)
    lab = np.empty(len(files), dtype=np.int64)
    keepsrc = []
    for i, (fp, sub) in enumerate(files):
        raw[i] = np.fromfile(fp, np.complex64, M)
        lab[i] = int(os.path.basename(fp).split('_')[1]) % N
        keepsrc.append(sub)
    est = decode_loraphy_batch(raw, P, upsampling)
    keep = est == lab
    stat = dict(total=int(len(files)), kept=int(keep.sum()),
                dropped=int((~keep).sum()), keep_rate=float(keep.mean()))
    if verbose:
        print(f'  필터: {stat["kept"]}/{stat["total"]} 통과 ({stat["keep_rate"]*100:.2f}%)')

    X, y = raw[keep], lab[keep]
    pk = np.array(keepsrc)[keep]
    if max_symbols and len(X) > max_symbols:
        sel = np.random.default_rng(0).choice(len(X), max_symbols, replace=False)
        X, y, pk = X[sel], y[sel], pk[sel]
        if verbose:
            print(f'  {max_symbols}개로 부분표집')
    if cache:
        np.savez(cache, X=X, y=y, pk=pk, stat=np.array(stat, dtype=object))
    return X, y, stat, pk


def add_noise_theirs(Y, snr_db, rng, mode='theirs', normalize=True, post=None):
    """그들 add_noise 를 그대로 옮긴다.

        amp = math.pow(0.1, snr/20) * torch.mean(torch.abs(dataY))   # 배치 스칼라
        noise = amp/sqrt(2) * (randn(num_samples) + 1j*randn(num_samples))  # (M,) 하나
        dataX = dataY + noise                                        # 배치에 브로드캐스트
        if normalization:
            dataX = dataX / torch.mean(torch.abs(dataX))             # 배치 스칼라

    ** 정규화는 반드시 켜야 한다. ** argmax 검출기(decode_loraphy, 표준 RX)는
    스케일에 불변이라 영향이 없지만, 신경망은 mean|x|=1 로 정규화된 입력에 학습됐다.
    정규화를 빼면 모델이 학습 분포 밖의 입력을 받아, 스케일이 우연히 맞는 좁은 SNR
    구간에서만 동작하고 나머지에서는 baseline 보다 나빠진다.

    mode='theirs'     : 위 코드 그대로 (배치 스칼라 amp, 배치 공유 잡음)
    mode='per-symbol' : 심볼별 독립 잡음 (통계적으로 낫지만 그들 코드와 다름)
    """
    if mode == 'theirs':
        amp = (10.0**(-snr_db/20.0))*np.abs(Y).mean()
        w = rng.standard_normal(Y.shape[1]) + 1j*rng.standard_normal(Y.shape[1])
        X = Y + (amp/np.sqrt(2))*w[None, :]
        if post is not None:          # LPF 는 정규화 '앞' — 학습과 순서가 같아야 한다
            X = post(X)
        if normalize:
            X = X/np.abs(X).mean()
    else:
        amp = (10.0**(-snr_db/20.0))*np.abs(Y).mean(1, keepdims=True)
        w = rng.standard_normal(Y.shape) + 1j*rng.standard_normal(Y.shape)
        X = Y + (amp/np.sqrt(2))*w
        if post is not None:
            X = post(X)
        if normalize:
            X = X/np.abs(X).mean(1, keepdims=True)
    return X.astype(np.complex64)


# -------------------------------------------------- B. DNN
def infer_sf(sd):
    """가중치 모양에서 체크포인트의 SF 를 역산한다.

    maskCNN:  lstm.weight_ih_l0 = [4*lstm_dim, conv_dim_lstm],  conv_dim_lstm = M = 8*2^SF
              fc2.weight        = [freq_size*2, fc1_dim],       freq_size     = N = 2^SF
    C_XtoY:   dense.weight      = [conv_dim_out*4, conv_dim_lstm*4]
    """
    for key, div in (('fc2.weight', 2), ('lstm.weight_ih_l0', None),
                     ('dense.weight', None)):
        if key not in sd:
            continue
        shp = tuple(sd[key].shape)
        n = shp[0]//div if div else (shp[1] if key == 'lstm.weight_ih_l0' else shp[1]//32)
        if key == 'lstm.weight_ih_l0':
            n = n//8                      # conv_dim_lstm = 8*N
        if n > 0 and (n & (n - 1)) == 0:  # 2 의 거듭제곱인가
            return int(math.log2(n))
    return None


def find_ckpt(ckpt_dir, pattern, explicit=None):
    """체크포인트를 찾는다. 반복수(100000)가 다를 수 있으므로 패턴으로 고른다."""
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f'지정한 체크포인트가 없다: {explicit}')
        return explicit
    cands = []
    if os.path.isdir(ckpt_dir):
        for fn in os.listdir(ckpt_dir):
            if pattern.lower() in fn.lower() and fn.endswith(('.pkl', '.pt', '.pth')):
                cands.append(os.path.join(ckpt_dir, fn))
    if not cands:
        here = (sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir)
                else f'(디렉터리 자체가 없음: {ckpt_dir})')
        raise FileNotFoundError(
            f"'{pattern}' 에 해당하는 체크포인트를 못 찾았다.\n"
            f"  찾아본 곳: {os.path.abspath(ckpt_dir)}\n"
            f"  그 안의 내용: {here}\n"
            f"  -> Google Drive 에서 받은 파일을 그 폴더에 두거나,\n"
            f"     --ckpt-mask / --ckpt-cls 로 경로를 직접 지정할 것.\n"
            f"     WSL 에서 찾기:  find ~ -iname '*maskCNN*' -o -iname '*C_XtoY*'")
    # 반복수가 큰 것을 고른다 (100000 > 90000)
    def it(p):
        base = os.path.basename(p)
        digits = ''.join(c for c in base.split('_')[0] if c.isdigit())
        return int(digits) if digits else -1
    cands.sort(key=it, reverse=True)
    if len(cands) > 1:
        print(f'  후보 {len(cands)}개 중 선택: {os.path.basename(cands[0])}')
    return cands[0]


class DNN:
    _torch = None

    """그들 test() 는 .eval() 을 부르지 않는다 — train() 안 226~227행에만 있다.

    모델에는 BatchNorm2d 8개와 Dropout(0.2)/(0.5) 가 있으므로, 그들 평가는
      - BatchNorm 이 running statistics 가 아니라 '테스트 배치 통계' 를 쓰고
      - Dropout 이 켜진 채로
    돌아간다. 이 체크포인트에서는 train 모드가 eval 모드보다 1.5 dB 낫다.
    Dropout 이 켜져 있는데도 그렇다는 것은 running BN 통계가 수렴하지 않았다는 뜻이다.

    mode='train' 이 그들 파이프라인의 충실한 재현이고, 'eval' 이 통상적으로 올바른
    추론 방식이다. 둘 다 잴 수 있게 해 둔다.
    """

    def __init__(self, sf, ckpt_dir, P, device='cpu', mask_path=None, cls_path=None,
                 mode='train'):
        import torch
        DNN._torch = torch
        from model_components import maskCNNModel, classificationHybridModel
        N, M = P['N'], P['M']
        self.N, self.M, self.dev = N, M, device
        self.mask = maskCNNModel(conv_dim_lstm=M, lstm_dim=400, fc1_dim=600, freq_size=N)
        self.cls = classificationHybridModel(conv_dim_in=2, conv_dim_out=N, conv_dim_lstm=M)
        for mdl, pat, exp in ((self.mask, 'maskCNN', mask_path),
                              (self.cls, 'C_XtoY', cls_path)):
            p = find_ckpt(ckpt_dir, pat, exp)
            try:                                  # torch>=2.6 은 weights_only 기본 True
                torch = DNN._torch
                sd = torch.load(p, map_location='cpu', weights_only=False)
            except TypeError:
                sd = torch.load(p, map_location='cpu')
            if hasattr(sd, 'state_dict'):         # 모델 통째로 저장된 경우
                sd = sd.state_dict()
            ck_sf = infer_sf(sd)
            if ck_sf is not None and ck_sf != sf:
                raise RuntimeError(
                    f'체크포인트가 SF{ck_sf} 용인데 SF{sf} 로 돌리고 있다.\n'
                    f'  파일: {p}\n'
                    f'  -> Google Drive 의 checkpoint/sf{sf}/ 에서 받거나,\n'
                    f'     --sf {ck_sf} 로 돌리고 --data-dir 도 SF{ck_sf} 데이터로 바꿀 것.')
            mdl.load_state_dict(sd, strict=True)
            mdl.to(device)
            mdl.train() if mode == 'train' else mdl.eval()
            print(f'  로드: {p}')
        print(f'  모델 모드: {mode}'
              + ('  (그들 test() 와 동일 — BatchNorm 이 배치 통계를 쓴다)'
                 if mode == 'train' else '  (통상적 추론)'))

    def stft(self, x):
        torch = DNN._torch
        """그들 perform_stft 그대로."""
        full = torch.stft(input=x, n_fft=self.M, hop_length=self.N//4,
                          win_length=self.N//2, pad_mode='constant', return_complex=True)
        img = torch.concat((full[:, -self.N//2:, :], full[:, 0:self.N//2, :]), axis=1)
        return torch.stack((img.real, img.imag), 1)

    def predict(self, Y):
        torch = DNN._torch
        with torch.no_grad():
            x = torch.tensor(Y, dtype=torch.cfloat)
            out = self.cls(self.mask(self.stft(x).to(self.dev)))
            return torch.max(out, 1)[1].cpu().numpy()


# -------------------------------------------------- 측정
def main(a):
    np.random.seed(10); random.seed(10)
    P = build_params(a.sf)
    print(f'SF{a.sf}  N={P["N"]}  M={P["M"]}  OSF={P["osf"]}')

    Xbank = chirp_bank_np(a.sf, 8)
    X, y, stat, pk = load_data(P, a.data_dir, a.cache, a.max_symbols, a.upsampling)
    X_all, y_all, held = X, y, None
    if a.only_idx:
        idx = np.load(a.only_idx)
        if len(X) != stat['kept']:
            raise SystemExit(
                '--only-idx 는 전체 집합에서만 쓸 수 있다. '
                f'지금 {len(X)}심볼인데 필터 통과분은 {stat["kept"]}심볼이다. '
                f'캐시가 부분표집본이므로 지우고 --max-symbols 0 으로 다시 만들 것: '
                f'rm {a.cache or "<cache>"}')
        held = np.zeros(len(X), dtype=bool); held[idx] = True
        X, y, pk = X[idx], y[idx], (pk[idx] if pk is not None else None)
        print(f'  held-out 만 평가: {a.only_idx} -> {len(X)}심볼')
    print(f'  평가 심볼 {len(X)}개')

    if a.selftest:
        print('\n[자체검사] 벡터화 decode_loraphy 가 원문과 같은가')
        ref = np.array([decode_loraphy_scalar(X[i], P['N'], P['downchirp'], a.upsampling)
                        for i in range(min(64, len(X)))])
        got = decode_loraphy_batch(X[:len(ref)], P, a.upsampling)
        print(f'  일치 {int((ref == got).sum())}/{len(ref)}'
              + ('  OK' if (ref == got).all() else '  ** 불일치! 여기서 멈출 것'))
        if not (ref == got).all():
            return

    dnn = (None if a.no_dnn else
           DNN(a.sf, a.ckpt_dir, P, a.device, a.ckpt_mask, a.ckpt_cls,
               a.model_mode))
    base_n = base_chirp_n(a.sf)
    post_fn = (lambda Z: brickwall_lpf_np(Z, P)) if a.lpf else None

    # --- 판정 기준선: 정렬된 MF. 정리가 예측하는 상한은 표준 RX 가 아니라 D3 이다.
    #     DNN 은 잔여 CFO/타이밍을 학습으로 보정할 수 있고 그것은 정리 위반이 아니다.
    Xn = Xbank/np.linalg.norm(Xbank, axis=1, keepdims=True)
    ref = {}
    if a.refs:
        if held is not None:                      # held-out 밖 심볼로 프로토타입
            Xp, yp = X_all[~held], y_all[~held]
        else:
            Xp, yp = X, y
        g_tau, g_eps = align_from_labels_np(Xp.astype(np.complex128), yp, Xbank)
        print(f'  기준선 정렬: tau={g_tau:+.2f} 샘플, eps={g_eps:+.4f} bin  '
              f'(프로토타입 {len(Xp)}심볼)')
        Pal = apply_corr_np(Xp.astype(np.complex128), g_tau, g_eps)
        by = {c: [] for c in range(P['N'])}
        for yv, c in zip(Pal, yp):
            if len(by[c]) < 30:
                by[c].append(yv)
        Pemp = empirical_prototypes_np(by, P['N'], P['M'])
        cov = float(np.mean([len(v) > 0 for v in by.values()]))
        print(f'  경험 프로토타입 코드 커버리지 {cov*100:.0f}%'
              + ('' if cov > 0.95 else '  ** 부족: D3 는 참고값 **'))
        ref = {'D2_aligned': (lambda Y: d_mf_np(apply_corr_np(Y.astype(np.complex128), g_tau, g_eps), Xn)),
               'D3_aligned': (lambda Y: d_mf_np(apply_corr_np(Y.astype(np.complex128), g_tau, g_eps), Pemp))}
    if a.lpf:
        print('  입력에 ±BW/2 브릭월 LPF 적용 (잡음 -> LPF -> 정규화, 학습과 동일 순서)')
    snrs = [float(v) for v in a.snrs.split(',')] if a.snrs else list(range(-30, 1))

    # ================= 2단계 측정 =================
    # 1단계(비쌈): 잡음 실현 R 개만 뽑아 '심볼별 정오답' 을 한 번만 계산해 둔다.
    # 2단계(거의 공짜): 패킷 재표집은 그 배열을 인덱싱만 한다. 재계산이 없으므로
    #                   B=1000 복제도 몇 초다.
    # §5-7 에서 잡음 재추출이 구간을 넓히지 않음을 확인했으므로 R 은 작아도 된다.
    # 구간을 정하는 것은 패킷 간 편차다.
    arms = (['decode_loraphy', 'standard_rx']
            + ([] if dnn is None else ['nelora_dnn']) + list(ref))
    R = max(1, a.noise_reals)
    n = len(X)
    ok_arr = {k: np.zeros((R, len(snrs), n), dtype=bool) for k in arms}

    print(f'\n[1단계] 잡음 실현 {R}개 × SNR {len(snrs)}점 × 팔 {len(arms)}개'
          f'  (심볼 {n}, 배치 {a.batch})')
    import time as _time
    _t0 = _time.time()
    for r in range(R):
        rng = np.random.default_rng(1000 + r)
        if dnn is not None and DNN._torch is not None:
            # train 모드라 Dropout 이 켜져 있다. 시드를 고정해야 재현되고,
            # 그 무작위성이 '잡음 실현' 의 일부로 들어간다.
            DNN._torch.manual_seed(1000 + r)
        for j, s in enumerate(snrs):
            for i in range(0, n, a.batch):
                Yb, lb = X[i:i+a.batch], y[i:i+a.batch]
                # 배치 크기·구성은 그들 것을 그대로 둔다. train 모드 BatchNorm 이
                # 테스트 배치 통계를 쓰므로 배치를 키우면 결과가 바뀐다.
                Yn = add_noise_theirs(Yb, s, rng, a.noise_mode, not a.no_norm, post_fn)
                pred = {'decode_loraphy': decode_loraphy_batch(Yn, P, a.upsampling),
                        'standard_rx': standard_rx(Yn, P, base_n)}
                for _k, _f in ref.items():
                    pred[_k] = _f(Yn)
                if dnn is not None:
                    pred['nelora_dnn'] = dnn.predict(Yn)
                for k in arms:
                    ok_arr[k][r, j, i:i+len(lb)] = (pred[k] == lb)
        el = _time.time() - _t0
        print(f'  실현 {r+1}/{R}  경과 {el/60:.1f}분'
              + (f'  남은 예상 {el/(r+1)*(R-r-1)/60:.1f}분' if r + 1 < R else ''),
              flush=True)

    # --- 본 표: R 개 실현 평균
    codes = np.unique(y)
    by_code = [np.where(y == c)[0] for c in codes]
    res = {k: {} for k in arms}
    for j, s in enumerate(snrs):
        for k in arms:
            micro = float(ok_arr[k][:, j, :].mean())
            macro = float(np.mean([ok_arr[k][:, j, ix].mean() for ix in by_code]))
            res[k][s] = dict(micro_acc=micro, macro_acc=macro)
        print(f'  SNR {s:+6.1f}  ' + '  '.join(
            f'{k}={res[k][s]["micro_acc"]*100:5.1f}%' for k in arms))

    base_out = dict(sf=a.sf, n_symbols=int(n), filter_stat=stat, bootstrap=None,
                    upsampling=a.upsampling, snrs=snrs, results=res,
                    noise_mode=a.noise_mode, normalization=(not a.no_norm),
                    model_mode=a.model_mode, lpf=bool(a.lpf), noise_reals=R,
                    note='held-out; 부트스트랩 전 중간 저장')
    with open(a.out, 'w') as f:
        json.dump(base_out, f, indent=2)
    print(f'\n중간 저장: {a.out}')

    # --- 2단계: 패킷 재표집 (재계산 없음)
    boot_out = None
    if a.boot and pk is not None:
        print(f'[2단계] 패킷 재표집 {a.boot} 복제 — 저장된 정오답 인덱싱만')
        upk, inv = np.unique(pk, return_inverse=True)
        idx_by = [np.where(inv == i)[0] for i in range(len(upk))]
        rgb = np.random.default_rng(7)
        pairs = [(x, z) for x in arms for z in arms if x != z]
        want = [('standard_rx', 'nelora_dnn'), ('decode_loraphy', 'nelora_dnn'),
                ('decode_loraphy', 'standard_rx'), ('D3_aligned', 'nelora_dnn'),
                ('decode_loraphy', 'D3_aligned')]
        want = [w for w in want if w[0] in arms and w[1] in arms]
        acc_g = {w: [] for w in want}
        for b in range(a.boot):
            r = int(rgb.integers(R))
            sel = np.concatenate([idx_by[i] for i in rgb.integers(0, len(upk), len(upk))])
            cr = {}
            for k in arms:
                ser = [(1 - ok_arr[k][r, j, sel].mean())*100 for j in range(len(snrs))]
                cr[k] = cross_np(list(snrs), ser, 10.0)
            for w in want:
                if cr[w[0]] is not None and cr[w[1]] is not None:
                    acc_g[w].append(cr[w[0]] - cr[w[1]])
        boot_out = {}
        print(f'  {"격차":<34}{"중앙값":>10}{"95% CI":>22}')
        for w in want:
            v = acc_g[w]
            if not v:
                continue
            arr = np.array(v)
            lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
            med = float(np.median(arr))
            boot_out[f'{w[0]}_minus_{w[1]}'] = dict(median=med, lo=lo, hi=hi, n=len(arr))
            sig = '유의' if (lo > 0 or hi < 0) else '0 과 구별 안 됨'
            print(f'  {w[0]} − {w[1]:<20}{med:>+9.2f}  [{lo:+6.2f}, {hi:+6.2f}]  {sig}')
        base_out['bootstrap'] = boot_out
        base_out['note'] = 'held-out; 2단계 부트스트랩 (잡음 실현 R, 패킷 재표집 B)'
        with open(a.out, 'w') as f:
            json.dump(base_out, f, indent=2)

    if dnn is not None:
        hi = [v for v in snrs if v >= 0]
        if hi:
            print('\n[온전성] 고SNR 에서 DNN 이 100% 에 가까워야 한다')
            for v in hi:
                print(f'  SNR {v:+5.0f}  DNN {res["nelora_dnn"][v]["micro_acc"]*100:6.2f}%'
                      f'   baseline {res["decode_loraphy"][v]["micro_acc"]*100:6.2f}%'
                      f'   표준RX {res["standard_rx"][v]["micro_acc"]*100:6.2f}%')
            worst = min(res['nelora_dnn'][v]['micro_acc'] for v in hi)
            if worst < 0.97:
                print('  ** 고SNR 에서도 100%% 가 아니다 (최저 %.2f%%). 파이프라인을 의심할 것.'
                      % (worst*100))

    print(f'\n저장: {a.out}  <- 이 파일만 가져오면 된다')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sf', type=int, default=7)
    ap.add_argument('--data-dir', required=True, help='NeLoRa_Dataset/<sf>/ 경로')
    ap.add_argument('--ckpt-dir', default='checkpoint/sf7')
    ap.add_argument('--ckpt-mask', default=None, help='maskCNN 가중치 경로 직접 지정')
    ap.add_argument('--ckpt-cls', default=None, help='C_XtoY 가중치 경로 직접 지정')
    ap.add_argument('--out', default='result.json')
    ap.add_argument('--cache', default='')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--upsampling', type=int, default=100)
    ap.add_argument('--max-symbols', type=int, default=3000,
                    help='부분표집 (0 이면 전체). 전체는 매우 느리다')
    ap.add_argument('--snrs', type=str, default='10,5,0,-5,-10,-12,-14,-15,-16,-17,-18,-19,-20,-22,-24',
                    help='10%% 교차 구간만 보면 충분하다')
    ap.add_argument('--no-dnn', action='store_true', help='체크포인트 없이 baseline 만')
    ap.add_argument('--refs', action='store_true', default=True,
                    help='정렬된 D2/D3 기준선도 함께 측정 (판정 상한)')
    ap.add_argument('--no-refs', dest='refs', action='store_false')
    ap.add_argument('--only-idx', default=None,
                    help='split_sf<SF>_test.npy — held-out 인덱스만 평가 (--max-symbols 0 필요)')
    ap.add_argument('--lpf', action='store_true',
                    help='입력에 ±BW/2 브릭월 LPF 적용 (LPF 로 학습한 모델 평가용)')
    ap.add_argument('--boot', type=int, default=1000,
                    help='패킷 재표집 복제 수. 2단계라 거의 공짜다')
    ap.add_argument('--noise-reals', type=int, default=10,
                    help='잡음 실현 수 R. 1단계 비용이 여기 비례한다')
    ap.add_argument('--boot-u', type=int, default=10,
                    help='부트스트랩용 upsampling (U=10 은 U=100 과 동치, 10배 빠름)')
    ap.add_argument('--model-mode', choices=['train', 'eval'], default='train',
                    help="train=그들 test() 와 동일(.eval() 미호출). eval=통상적 추론")
    ap.add_argument('--noise-mode', choices=['theirs', 'per-symbol'], default='theirs',
                    help="theirs=그들 코드 그대로(배치 공유 잡음)")
    ap.add_argument('--no-norm', action='store_true',
                    help='정규화 끄기 — 진단용. DNN 을 학습 분포 밖으로 내몬다')
    ap.add_argument('--selftest', action='store_true', default=True)
    a = ap.parse_args()
    if a.max_symbols == 0:
        a.max_symbols = None
    main(a)
