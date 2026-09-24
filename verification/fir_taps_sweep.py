# -*- coding: utf-8 -*-
"""§8 한계 4번에 답한다 — 브릭월(이상 필터) 대신 실현 가능한 유한 FIR 을 쓰면
원본 LoRaPHY 팔이 얼마를 잃는가.

원본 MATLAB 의 lowpass() 는 최소차수 설계(steepness 0.85, 저지대역 60 dB)이고
탭 수를 우리가 고를 수 없다. 그래서 "원본의 필터" 를 흉내내는 대신, **설계를
바꿔가며 결론이 필터 설계에 얼마나 의존하는지**를 잰다. 같은 held-out · 같은
잡음 규약(배치 16, mode=theirs) 이다.

  python fir_taps_sweep.py
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.signal import firwin, lfilter
from loraphy_orig import decode_loraphy_orig, lpf_brick
from nelora_dnn_eval import add_noise_theirs

SF, OSF, N, B = 7, 8, 128, 16
R = 4
SNRS = [-10, -14, -16, -18, -20, -22]
TAPS = [17, 33, 65, 129, 513]
here = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(here, 'results/cache_sf7.npz'), allow_pickle=True)
idx = np.load(os.path.join(here, 'results/split_sf7_test.npy'))
X, y = d['X'][idx], d['y'][idx]


def fir_arm(nt):
    h = firwin(nt, cutoff=1.0/OSF)          # 통과대역 끝 BW/2
    dd = (nt - 1)//2                        # 선형위상 군지연
    def f(Y):
        p = np.concatenate([Y, np.zeros((len(Y), dd), Y.dtype)], axis=1)
        Z = lfilter(h, 1.0, p, axis=1)[:, dd:]
        return decode_loraphy_orig(Z, SF, OSF, 10, 'raw')
    return f


ARMS = {'브릭월': lambda Y: decode_loraphy_orig(lpf_brick(Y, OSF, N), SF, OSF, 10, 'raw')}
ARMS.update({f'FIR{nt}': fir_arm(nt) for nt in TAPS})

print(f'SF{SF} held-out {len(X)}심볼 · 배치 {B} · 실현 {R} · firwin(Hamming), 차단 BW/2\n')
print('   SNR  ' + '  '.join(f'{k:>8}' for k in ARMS))
acc = {k: [] for k in ARMS}
for s in SNRS:
    per = {k: [] for k in ARMS}
    for r in range(R):
        rng = np.random.default_rng(1000 + r)
        ok = {k: np.zeros(len(X), bool) for k in ARMS}
        for i in range(0, len(X), B):
            Yn = add_noise_theirs(X[i:i+B], s, rng, 'theirs', True, None)
            for k, f in ARMS.items():
                ok[k][i:i+len(Yn)] = (f(Yn) == y[i:i+len(Yn)])
        for k in ARMS:
            per[k].append(ok[k].mean())
    for k in ARMS:
        acc[k].append(float(np.mean(per[k])))
    print(f'  {s:+5.1f} ' + '  '.join(f'{acc[k][-1]*100:7.2f}%' for k in ARMS))


def snr_at(k, t):
    x, v = np.array(SNRS, float), np.array(acc[k])
    o = np.argsort(x)
    return float(np.interp(t, v[o], x[o])) if v.min() <= t <= v.max() else float('nan')


print('\n  10%/20%/30% SER 교차점과 브릭월 대비 (음수 = FIR 이 나음)')
for t, nm in ((0.90, '10%'), (0.80, '20%'), (0.70, '30%')):
    ref = snr_at('브릭월', t)
    print(f'   {nm} SER  브릭월={ref:+6.2f}  ' + '  '.join(
        f'{k}={snr_at(k,t)-ref:+5.2f}' for k in ARMS if k != '브릭월'))
