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
import torch
from scipy.signal import chirp
from scipy.fft import fft


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
        Z = np.fft.fft(Y[i:i+chunk]*down[None, :], M*upsampling, axis=1)
        s = np.abs(Z[:, :T]) + np.abs(Z[:, -T:])
        out[i:i+chunk] = np.round(np.argmax(s, 1)/upsampling).astype(np.int64) % N
    return out


# -------------------------------------------------- C. 표준 수신기
def base_chirp_n(sf):
    """해석적 2차 위상, OSF=1 (N 샘플). 그들 downchirp 과 상수 위상만 다르다."""
    N = 2**sf
    n = np.arange(N)
    return np.exp(2j*np.pi*(n**2/(2.0*N) - n/2.0))


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
            return d['X'], d['y'], d['stat'].item()

    N, M = P['N'], P['M']
    files = []
    for sub in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, sub)
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                files.append((os.path.join(p, fn), int(fn.split('_')[1]), sub))
    if verbose:
        print(f'  파일 {len(files)}개 발견')

    X, y, pk = [], [], []
    raw = np.empty((len(files), M), dtype=np.complex64)
    lab = np.empty(len(files), dtype=np.int64)
    keepsrc = []
    for i, (fp, truth, sub) in enumerate(files):
        a = np.fromfile(fp, np.complex64, M)
        if a.size != M:
            continue
        raw[i] = a; lab[i] = truth; keepsrc.append(sub)
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
    return X, y, stat


def add_noise_theirs(Y, snr_db, rng):
    """그들 add_noise: 평균 진폭 기준. 정규화는 argmax 에 영향 없어 생략."""
    amp = (10.0**(-snr_db/20.0))*np.abs(Y).mean(1, keepdims=True)
    n = rng.standard_normal(Y.shape) + 1j*rng.standard_normal(Y.shape)
    return (Y + (amp/np.sqrt(2))*n).astype(np.complex64)


# -------------------------------------------------- B. DNN
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
    def __init__(self, sf, ckpt_dir, P, device='cpu', mask_path=None, cls_path=None):
        from model_components import maskCNNModel, classificationHybridModel
        N, M = P['N'], P['M']
        self.N, self.M, self.dev = N, M, device
        self.mask = maskCNNModel(conv_dim_lstm=M, lstm_dim=400, fc1_dim=600, freq_size=N)
        self.cls = classificationHybridModel(conv_dim_in=2, conv_dim_out=N, conv_dim_lstm=M)
        for mdl, pat, exp in ((self.mask, 'maskCNN', mask_path),
                              (self.cls, 'C_XtoY', cls_path)):
            p = find_ckpt(ckpt_dir, pat, exp)
            try:                                  # torch>=2.6 은 weights_only 기본 True
                sd = torch.load(p, map_location='cpu', weights_only=False)
            except TypeError:
                sd = torch.load(p, map_location='cpu')
            if hasattr(sd, 'state_dict'):         # 모델 통째로 저장된 경우
                sd = sd.state_dict()
            mdl.load_state_dict(sd, strict=True)
            mdl.to(device).eval()
            print(f'  로드: {p}')

    def stft(self, x):
        """그들 perform_stft 그대로."""
        full = torch.stft(input=x, n_fft=self.M, hop_length=self.N//4,
                          win_length=self.N//2, pad_mode='constant', return_complex=True)
        img = torch.concat((full[:, -self.N//2:, :], full[:, 0:self.N//2, :]), axis=1)
        return torch.stack((img.real, img.imag), 1)

    @torch.no_grad()
    def predict(self, Y):
        x = torch.tensor(Y, dtype=torch.cfloat)
        out = self.cls(self.mask(self.stft(x).to(self.dev)))
        return torch.max(out, 1)[1].cpu().numpy()


# -------------------------------------------------- 측정
def main(a):
    np.random.seed(10); random.seed(10); torch.manual_seed(10)
    P = build_params(a.sf)
    print(f'SF{a.sf}  N={P["N"]}  M={P["M"]}  OSF={P["osf"]}')

    X, y, stat = load_data(P, a.data_dir, a.cache, a.max_symbols, a.upsampling)
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
           DNN(a.sf, a.ckpt_dir, P, a.device, a.ckpt_mask, a.ckpt_cls))
    base_n = base_chirp_n(a.sf)
    snrs = [float(v) for v in a.snrs.split(',')] if a.snrs else list(range(-30, 1))

    arms = ['decode_loraphy', 'standard_rx'] + ([] if dnn is None else ['nelora_dnn'])
    res = {k: {} for k in arms}
    rng = np.random.default_rng(1)
    print(f'\nSNR {snrs[0]:.0f} ~ {snrs[-1]:.0f} ({len(snrs)}점), 팔 {len(arms)}개')
    for s in snrs:
        hit = {k: 0 for k in arms}
        per = {k: {} for k in arms}          # 코드별 (macro 평균용)
        for i in range(0, len(X), a.batch):
            Yb, lb = X[i:i+a.batch], y[i:i+a.batch]
            Yn = add_noise_theirs(Yb, s, rng)          # 세 팔이 같은 잡음을 본다
            pred = {'decode_loraphy': decode_loraphy_batch(Yn, P, a.upsampling),
                    'standard_rx': standard_rx(Yn, P, base_n)}
            if dnn is not None:
                pred['nelora_dnn'] = dnn.predict(Yn)
            for k in arms:
                ok = pred[k] == lb
                hit[k] += int(ok.sum())
                for c, o in zip(lb, ok):
                    d = per[k].setdefault(int(c), [0, 0])
                    d[0] += int(o); d[1] += 1
        for k in arms:
            micro = hit[k]/len(X)
            macro = float(np.mean([v[0]/v[1] for v in per[k].values()]))
            res[k][s] = dict(micro_acc=micro, macro_acc=macro)
        print(f'  SNR {s:+6.1f}  ' + '  '.join(
            f'{k}={res[k][s]["micro_acc"]*100:5.1f}%' for k in arms))

    out = dict(sf=a.sf, n_symbols=int(len(X)), filter_stat=stat,
               upsampling=a.upsampling, snrs=snrs, results=res,
               note='as-run: DNN saw ~90% of these symbols in training '
                    '(train/test split not reproducible; torch seed unset)')
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=2)
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
    ap.add_argument('--snrs', type=str, default='-10,-12,-14,-15,-16,-17,-18,-19,-20,-22,-24',
                    help='10%% 교차 구간만 보면 충분하다')
    ap.add_argument('--no-dnn', action='store_true', help='체크포인트 없이 baseline 만')
    ap.add_argument('--selftest', action='store_true', default=True)
    a = ap.parse_args()
    if a.max_symbols == 0:
        a.max_symbols = None
    main(a)
