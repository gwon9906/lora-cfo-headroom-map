# -*- coding: utf-8 -*-
"""
그들 main.py 로부터 '공정하게 비교 가능한' 학습 스크립트를 만든다.

바꾸는 것 네 가지. 전부 학습 조건을 통제하기 위한 것이고, 모델·손실·최적화기는
그들 것을 그대로 둔다.

  1. --lpf : add_noise() 끝에서 dataX·dataY 에 ±BW/2 브릭월 LPF 를 건다.
       순서는 **깨끗한 신호 -> 잡음 -> LPF -> 정규화**.
       정규화를 LPF 앞에 두면 입력 스케일이 달라진다(LPF 가 잡음 7/8 을 지우므로
       mean|x| 가 크게 바뀐다). 평가 하네스(nelora_dnn_eval --lpf)도 같은 순서다.
       OSF=8 과 텐서 모양은 유지한다 — 데시메이션까지 하면 입력 크기가 N 으로 줄어
       아키텍처를 고쳐야 하므로, 대역 밖 잡음만 지우고 표본화율은 그대로 둔다.

  2. 시드 고정 + **패킷 단위** 분할을 파일로 저장.
       그들 random_split 은 torch 시드를 잡지 않아 분할 복원이 불가능했다(그래서
       held-out 측정을 못 했다). 여기서는 패킷(폴더)을 9:1 로 나누고 인덱스를 남긴다.
       같은 패킷의 심볼이 양쪽에 들어가지 않으므로 그들 심볼 단위 분할보다 엄격하다.

  3. 학습곡선 CSV.
       "DNN 에게 충분한 기회를 줬다" 를 보이려면 검증 정확도가 평탄해진 것을 보여야
       한다. 귀무 결과는 학습을 덜 시켜도 쉽게 나온다.

  4. --scratch : 공개 체크포인트에서 이어서가 아니라 처음부터 학습.
       LPF 입력은 분포가 달라서 이어 학습이 불리할 수 있다.

사용법 (WSL, 그들 저장소 안에서):
  python nelora_lpf_patch.py main.py            # main_fair.py 생성

  # 두 팔을 같은 조건으로 (같은 시드·분할·에포크·프로세스 하나씩 순차)
  python main_fair.py --sf 7 --batch_size 64 --epochs 150            # 필터 없음
  python main_fair.py --sf 7 --batch_size 64 --epochs 150 --lpf      # 필터 입력

  # 평가 (held-out 만)
  python nelora_dnn_eval.py --sf 7 --ckpt-dir ckpt_sf7_plain \\
      --data-dir ~/LoRa/data/7/ --cache ~/LoRa/cache_sf7.npz \\
      --only-idx split_sf7_test.npy --boot 100 --out result_plain.json
  python nelora_dnn_eval.py --sf 7 --ckpt-dir ckpt_sf7_lpf --lpf \\
      --data-dir ~/LoRa/data/7/ --cache ~/LoRa/cache_sf7.npz \\
      --only-idx split_sf7_test.npy --boot 100 --out result_lpf.json
"""
import io, sys, os

HEADER = '''
# ============ 공정 비교용 추가 (nelora_lpf_patch.py 생성) ============
import csv as _csv
import argparse as _argparse

_extra = _argparse.ArgumentParser(add_help=False)
_extra.add_argument('--lpf', action='store_true')
_extra.add_argument('--epochs', type=int, default=None)
_extra.add_argument('--scratch', action='store_true')
_extra.add_argument('--seed', type=int, default=10)
_xopts, _ = _extra.parse_known_args()

USE_LPF = _xopts.lpf
FROM_SCRATCH = _xopts.scratch
SEED = _xopts.seed

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


def brickwall_lpf(x):
    """+-BW/2 브릭월. OSF 와 텐서 모양은 유지, 대역 밖 잡음만 지운다."""
    X = torch.fft.fft(x, dim=-1)
    keep = num_classes // 2
    mask = torch.zeros(num_samples, dtype=torch.bool, device=x.device)
    mask[:keep] = True
    mask[-(num_classes - keep):] = True
    return torch.fft.ifft(X * mask, dim=-1)
# ====================================================================
'''


def main(src, dst=None):
    dst = dst or os.path.join(os.path.dirname(src) or '.', 'main_fair.py')
    s = io.open(src, encoding='utf-8').read()

    # --- 1) 헤더 삽입 (상수 정의 뒤, 모델 생성 앞)
    anchor = "# define models"
    assert anchor in s, 'main.py 구조가 예상과 다르다 (define models)'
    s = s.replace(anchor, HEADER.strip('\n') + '\n\n' + anchor, 1)

    # --- 2) 체크포인트 디렉터리 분리
    s = s.replace("save_ckpt_dir = f'ckpt_sf{sf}'",
                  "save_ckpt_dir = f'ckpt_sf{sf}_' + ('lpf' if USE_LPF else 'plain')")

    # --- 3) scratch 면 사전학습 가중치 로드를 건너뛴다
    for nm in ('mask_CNN', 'C_XtoY'):
        old = f"{nm}.load_state_dict(torch.load({nm}_load_path"
        i = s.index(old)
        j = s.index('\n', i)
        line = s[i:j]
        s = s[:i] + f"if not FROM_SCRATCH:\n    " + line + s[j:]

    # --- 4) add_noise 에 LPF (정규화 '앞')
    old = """    if normalization:
        dataX = dataX / torch.mean(torch.abs(dataX)) # normalization
    return dataX, dataY, truth_idx"""
    new = """    if USE_LPF:                       # 잡음 -> LPF -> 정규화 (평가와 같은 순서)
        dataX = brickwall_lpf(dataX)
        dataY = brickwall_lpf(dataY)
    if normalization:
        dataX = dataX / torch.mean(torch.abs(dataX)) # normalization
    return dataX, dataY, truth_idx"""
    assert old in s, 'add_noise 구조가 예상과 다르다'
    s = s.replace(old, new, 1)

    # --- 5) load_data 가 패킷(폴더)도 돌려주게
    s = s.replace("""    files = [[] for i in range(num_classes)]
    for subfolder in os.listdir(data_dir):
        for filename in os.listdir(os.path.join(data_dir, subfolder)):
            truth_idx = int(filename.split('_')[1])
            files[truth_idx].append(os.path.join(data_dir, subfolder, filename))""",
"""    files = [[] for i in range(num_classes)]
    for subfolder in sorted(os.listdir(data_dir)):
        for filename in sorted(os.listdir(os.path.join(data_dir, subfolder))):
            truth_idx = int(filename.split('_')[1])
            files[truth_idx].append((os.path.join(data_dir, subfolder, filename), subfolder))""")
    s = s.replace("""    datax = []  # chirp symbols
    datay = []  # truth indexes for each symbol""",
"""    datax = []  # chirp symbols
    datay = []  # truth indexes for each symbol
    datapk = []  # packet (subfolder) for each symbol""")
    s = s.replace("""        for filepath in filelist:
            with open(filepath, 'rb') as fid:""",
"""        for filepath, _pk in filelist:
            with open(filepath, 'rb') as fid:""")
    s = s.replace("""                    datax.append(torch.tensor(chirp_raw, dtype=torch.cfloat))
                    datay.append(truth_idx)""",
"""                    datax.append(torch.tensor(chirp_raw, dtype=torch.cfloat))
                    datay.append(truth_idx)
                    datapk.append(_pk)""")
    s = s.replace("""    with open(f'pkl_{sf}.pkl', 'wb') as g:
        pickle.dump((datax, datay), g)

    return datax, datay""",
"""    with open(f'pkl_{sf}.pkl', 'wb') as g:
        pickle.dump((datax, datay, datapk), g)

    return datax, datay, datapk""")
    s = s.replace("""        with open(f'pkl_{sf}.pkl', 'rb') as g:
            datax, datay = pickle.load(g)
        return datax, datay""",
"""        with open(f'pkl_{sf}.pkl', 'rb') as g:
            _c = pickle.load(g)
        return _c if len(_c) == 3 else (_c[0], _c[1], None)""")

    # --- 6) train(): 패킷 단위 분할 + 인덱스 저장 + 학습곡선
    s = s.replace("""    datax, datay = load_data()
    data = TensorDataset(torch.stack(datax), torch.tensor(datay, dtype=torch.long))

# Calculate class weights for imbalanced datasets""",
"""    datax, datay, datapk = load_data()
    data = TensorDataset(torch.stack(datax), torch.tensor(datay, dtype=torch.long))

# Calculate class weights for imbalanced datasets""")
    s = s.replace("""# Split the dataset into 9:1 ratio
    train_size = int(0.9 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = random_split(data, [train_size, test_size])""",
"""# Split by PACKET (not by symbol): 같은 패킷이 양쪽에 들어가지 않는다
    _split_file = f'split_sf{sf}_test.npy'
    _pk = np.array(datapk)
    _upk = np.array(sorted(set(_pk.tolist())))
    _rs = np.random.RandomState(SEED)
    _perm = _rs.permutation(len(_upk))
    _n_test_pk = max(1, int(round(0.1 * len(_upk))))
    _test_pk = set(_upk[_perm[:_n_test_pk]].tolist())
    _test_idx = np.array([i for i, q in enumerate(_pk) if q in _test_pk])
    _train_idx = np.array([i for i, q in enumerate(_pk) if q not in _test_pk])
    np.save(_split_file, _test_idx)
    print(f'패킷 단위 분할: 학습 {len(_upk)-_n_test_pk}패킷/{len(_train_idx)}심볼, '
          f'검증 {_n_test_pk}패킷/{len(_test_idx)}심볼  -> {_split_file}')
    from torch.utils.data import Subset
    train_dataset = Subset(data, _train_idx.tolist())
    test_dataset = Subset(data, _test_idx.tolist())
    train_dataset.indices = _train_idx.tolist()
    test_dataset.indices = _test_idx.tolist()""")

    # --- 7) 에포크 수 오버라이드 + 학습곡선 CSV
    s = s.replace("    for epoch in range(train_epochs):",
                  "    _n_ep = _xopts.epochs or train_epochs\n"
                  "    _curve = open(f'curve_sf{sf}_' + ('lpf' if USE_LPF else 'plain') + '.csv', 'w', newline='')\n"
                  "    _cw = _csv.writer(_curve); _cw.writerow(['epoch', 'test_acc'])\n"
                  "    for epoch in range(_n_ep):")
    s = s.replace("""            print('SNR: %d TEST ACC: %.3f' % (test_snr, correct_count / (len(testing_loader) * batch_size)))""",
"""            _acc = float(correct_count) / max(1, len(test_dataset))
            print('SNR: %d TEST ACC: %.3f' % (test_snr, _acc))
            _cw.writerow([epoch, round(_acc, 5)]); _curve.flush()""")

    # --- 8) 학습만
    i = s.rindex("if __name__ == '__main__':")
    s = s[:i] + "if __name__ == '__main__':\n    train()\n"

    io.open(dst, 'w', encoding='utf-8').write(s)
    print(f'생성: {dst}')
    for line in ('--lpf 로 ±BW/2 브릭월 (잡음 -> LPF -> 정규화)',
                 '시드 고정 + 패킷 단위 9:1 분할, split_sf<SF>_test.npy 로 저장',
                 '학습곡선 curve_sf<SF>_{plain,lpf}.csv',
                 '체크포인트 ckpt_sf<SF>_{plain,lpf}/',
                 '--scratch 로 처음부터, --epochs 로 길이 지정'):
        print('  - ' + line)
    print('\n캐시 pkl_<SF>.pkl 은 패킷 정보가 추가돼 형식이 바뀌었으니 한 번 지울 것.')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'main.py')
