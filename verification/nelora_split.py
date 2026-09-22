# -*- coding: utf-8 -*-
"""
held-out 패킷 분할을 torch 없이 복원한다.

`nelora_lpf_patch.py` 가 그들 `main.py` 에 심는 분할과 **같은 규칙**이다 —
`load_data()` 가 돌려주는 패킷 id 를 정렬해 `RandomState(seed)` 로 섞고 앞의 10% 를
검증 패킷으로 쓴다. 학습은 WSL(torch)에서 돌지만, 그 분할이 어느 심볼을 가리키는지는
여기서 numpy 만으로 되살릴 수 있어야 한다. 그래야 고전 팔을 로컬에서 같은 모집단으로
다시 잴 수 있다(예: SNR 격자를 −34 dB 까지 늘릴 때).

SF7 기준 정답: 184패킷 중 18패킷 / 1353심볼.

사용법:
  python nelora_split.py --data-dir ../NeLoRa_Dataset/7 --sf 7
"""
import argparse, importlib.util, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_eval_module():
    spec = importlib.util.spec_from_file_location(
        'nde', os.path.join(HERE, 'nelora_dnn_eval.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(a):
    m = load_eval_module()
    P = m.build_params(a.sf)
    X, y, stat, pk = m.load_data(P, a.data_dir, a.cache, None, a.upsampling)

    # ** nelora_lpf_patch.py 가 심는 분할과 동일해야 한다. **
    upk = np.array(sorted(set(pk.tolist())))
    perm = np.random.RandomState(a.seed).permutation(len(upk))
    n_test = max(1, int(round(0.1 * len(upk))))
    test_pk = set(upk[perm[:n_test]].tolist())
    idx = np.array([i for i, q in enumerate(pk) if q in test_pk])

    print(f'  전체 {len(upk)}패킷 / {stat["kept"]}심볼 (필터 통과)')
    print(f'  held-out {n_test}패킷 / {len(idx)}심볼')
    print('  패킷: ' + ' '.join(sorted(test_pk)))
    np.save(a.out, idx)
    print(f'-> {a.out}')
    if a.sf == 7 and a.seed == 10 and (len(idx) != 1353 or n_test != 18):
        raise SystemExit('SF7/seed10 인데 18패킷/1353심볼이 아니다 — 분할 규칙이 어긋났다')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sf', type=int, default=7)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--cache', default=os.path.join(HERE, 'results', 'cache_sf7.npz'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'split_sf7_test.npy'))
    ap.add_argument('--seed', type=int, default=10, help='main_fair.py 의 --seed 와 같아야 한다')
    ap.add_argument('--upsampling', type=int, default=100)
    main(ap.parse_args())
