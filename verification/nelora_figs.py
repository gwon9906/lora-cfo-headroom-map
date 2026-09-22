# -*- coding: utf-8 -*-
"""
측정 JSON 에서 논문용 그림을 만든다.

  fig1_ser.png        SER vs SNR — baseline / DNN / 표준 RX / D3        (held-out, 필터 없음)
  fig2_bandlimit.png  대역제한 입력 전후 — 격차가 사라지는 것을 보인다
  fig3_sf.png         SF7~10 격차

라벨은 영어로 쓴다(논문용이고 한글 폰트가 없는 환경이 많다).

사용법:
  python nelora_figs.py            # results/*.json 을 읽어 results/ 에 png 생성
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# 색: 고전=파랑 계열, 신경망=주황, 상한=회색 점선
C = {'decode_loraphy': '#4C72B0', 'nelora_dnn': '#DD8452',
     'standard_rx': '#55A868', 'D3_aligned': '#8C8C8C'}
LBL = {'decode_loraphy': 'decode_loraphy (theirs)', 'nelora_dnn': 'NELoRa DNN',
       'standard_rx': 'Standard RX (no training, no alignment)',
       'D3_aligned': 'Empirical MF, aligned (ceiling)'}


def load(fn):
    d = json.load(open(os.path.join(R, fn), encoding='utf-8'))
    snrs = sorted(d['snrs'])
    ser = {k: [(1 - d['results'][k][f'{s}']['micro_acc'])*100 for s in snrs]
           for k in d['results']}
    return snrs, ser


def style(ax, title):
    ax.axhline(10, ls='--', lw=1, color='#B0B0B0', zorder=1)
    ax.text(ax.get_xlim()[0], 10, ' 10% SER', va='bottom', ha='left',
            fontsize=8, color='#808080')
    ax.set_xlabel('SNR (dB, synthetic AWGN on captures)')
    ax.set_ylabel('Symbol error rate (%)')
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=.25, lw=.6)
    ax.set_ylim(0, 60)


def fig1():
    snrs, ser = load('dnn_sf7_heldout_plain.json')
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for k in ('decode_loraphy', 'nelora_dnn', 'standard_rx', 'D3_aligned'):
        if k not in ser:
            continue
        ax.plot(snrs, ser[k], 'o-' if k != 'D3_aligned' else '--',
                ms=3.5, lw=1.8 if k != 'D3_aligned' else 1.2,
                color=C[k], label=LBL[k], zorder=3)
    ax.set_xlim(-24, -8)
    style(ax, 'SF7, held-out packets — their conditions, same batch')
    ax.legend(fontsize=8, loc='upper right', framealpha=.95)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig1_ser.png'), dpi=200)
    print('  fig1_ser.png')


def fig2():
    s1, a = load('dnn_sf7_heldout_plain.json')          # 필터 없음
    s2, b = load('dnn_sf7_heldout_lpf.json')            # 대역제한
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, (snrs, ser, t) in zip(axes, [
            (s1, a, 'Full-band input'), (s2, b, 'Band-limited input (±BW/2)')]):
        for k in ('decode_loraphy', 'standard_rx'):
            ax.plot(snrs, ser[k], 'o-', ms=3.5, lw=1.8, color=C[k], label=LBL[k], zorder=3)
        ax.set_xlim(-24, -8)
        style(ax, t)
    axes[1].set_ylabel('')
    axes[1].legend(fontsize=8, loc='center right', framealpha=.95)
    for ax, txt, col in ((axes[0], 'gap  +3.33 dB' + chr(10) + '[+2.67, +4.02]', '#B03030'),
                         (axes[1], 'gap  +0.28 dB' + chr(10) + '[-0.14, +0.70]   n.s.', '#2E7D32')):
        ax.text(.03, .95, txt, transform=ax.transAxes, va='top', ha='left',
                fontsize=9.5, color=col, weight='bold',
                bbox=dict(fc='white', ec=col, lw=.8, alpha=.9, pad=4))
    fig.suptitle('Band-limiting every arm removes the gap — only decode_loraphy moves',
                 fontsize=11, y=.99)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig2_bandlimit.png'), dpi=200)
    print('  fig2_bandlimit.png')


def fig3():
    sf = [7, 8, 9, 10]
    g10 = [4.14, 3.52, 3.92, 3.50]
    g20 = [3.38, 3.21, 3.10, 3.03]
    g30 = [3.20, 3.10, 2.95, 2.92]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    w = .26
    x = np.arange(len(sf))
    for i, (g, lb, c) in enumerate([(g10, '10% SER', '#4C72B0'),
                                    (g20, '20% SER', '#7BA3D0'),
                                    (g30, '30% SER', '#AEC7E8')]):
        ax.bar(x + (i-1)*w, g, w, label=lb, color=c, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f'SF{s}' for s in sf])
    ax.set_ylabel('decode_loraphy − Standard RX  (dB)')
    ax.set_title('The gap is not specific to SF7\n(as-run, their filter, nobody aligned)',
                 fontsize=10)
    ax.grid(axis='y', alpha=.25, lw=.6); ax.set_axisbelow(True)
    ax.legend(fontsize=8); ax.set_ylim(0, 4.8)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig3_sf.png'), dpi=200)
    print('  fig3_sf.png')


if __name__ == '__main__':
    print('그림 생성:')
    fig1(); fig2(); fig3()
    print(f'-> {R}')
