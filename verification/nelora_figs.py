# -*- coding: utf-8 -*-
"""
측정 JSON 에서 논문용 그림을 만든다.

  fig1_ser.png        SER vs SNR — baseline / DNN / 표준 RX / D3        (held-out, 필터 없음)
  fig2_bandlimit.png  대역제한 입력 전후 — 격차가 어디서 오는지 보인다
  fig3_sf.png         SF7~10 격차

규칙 (4차 검토 반영):
  * **semilogy.** 판정이 일어나는 SER 0~20% 구간이 선형축에서는 아래 1/5 로 눌린다.
    통신 분야 SER 곡선의 관례이기도 하다. SER=0 인 점은 로그축에 못 찍으므로 뺀다.
  * **기준선 세 개를 서로 다른 부호로.** 10% 임계(연한 점선), 랜덤 추측 1−1/N(더 연한
    점선), D3(보라 일점쇄선). 회색 파선 하나가 두 가지를 뜻하던 것을 없앴다.
  * **교차점을 그림에 적는다.** 10% 선까지 드롭라인 + 값. 이 보고서의 결론이 그 네 숫자다.
  * **흑백 인쇄 대비** 색과 선 스타일을 같이 다르게 준다. 주인공(표준 RX)만 굵게.
  * 라벨에 해석을 박지 않는다 — `Empirical MF (aligned)`. 그것이 상한인 근거(충분통계량)는
    캡션과 본편 README §4.2 에 쓴다.

축은 SER 이 포화하는 −34 dB 까지다(랜덤 추측 1−1/128 = 99.2%). 고전 팔은
`ext_sf7_{plain,lpf}.json`, DNN 팔은 `ext_sf7_dnn.json` (torch 가 필요해 그들 저장소에서
돌린다). 없으면 −24 dB 까지만 그린다.

사용법:
  python nelora_figs.py            # results/*.json 을 읽어 results/ 에 png 생성
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
YLO, THR = 1e-3, 0.10

# 색 + 선 스타일을 같이 다르게 준다 (흑백 인쇄).
STY = {
    'decode_loraphy': dict(color='#2B5FA8', ls='-',  lw=1.4, label='decode_loraphy (theirs)'),
    'nelora_dnn':     dict(color='#DD8452', ls='--', lw=1.4, label='NELoRa DNN'),
    'standard_rx':    dict(color='#2E8B57', ls='-',  lw=2.2,
                           label='Standard RX (no training, no alignment)'),
    'D3_aligned':     dict(color='#7B52AB', ls='-.', lw=1.4, label='Empirical MF (aligned)'),
}
ORDER = ('decode_loraphy', 'nelora_dnn', 'standard_rx', 'D3_aligned')


def load(fn):
    """{key: (snrs, ser)} — SER 은 비율. 팔마다 격자가 다를 수 있다."""
    d = json.load(open(os.path.join(R, fn), encoding='utf-8'))
    snrs = sorted(d['snrs'])
    return {k: (snrs, [1 - d['results'][k][f'{s}']['micro_acc'] for s in snrs])
            for k in d['results']}


def have(fn):
    return os.path.exists(os.path.join(R, fn))


def crossing(xs, ys, thr=THR):
    """SER 이 thr 를 지나는 SNR. 마지막 교차를 쓴다(§5-11 의 cross_np 와 같은 규약)."""
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    cr = None
    for i in range(len(x) - 1):
        if (y[i] - thr) * (y[i+1] - thr) <= 0 and y[i] != y[i+1]:
            cr = x[i] + (thr - y[i]) * (x[i+1] - x[i]) / (y[i+1] - y[i])
    return cr


def plot_arms(ax, ser, keys, short=()):
    """곡선을 그리고 교차점 목록을 돌려준다. SER=0 은 로그축에 못 찍으므로 뺀다."""
    lo, cross = -8, []
    for k in keys:
        if k not in ser:
            continue
        xs, ys = ser[k]
        lo = min(lo, min(xs))
        st = dict(STY[k])
        if k in short:
            st['label'] += '  (measured to -24 dB)'
        m = np.array(ys) > 0
        ax.plot(np.array(xs)[m], np.array(ys)[m], marker='o', ms=3, mew=0,
                zorder=3, **st)
        c = crossing(xs, ys)
        if c is not None:
            cross.append((c, STY[k]['color']))
    return lo, cross


def style(ax, title, xlo, n_classes=128, guess=True):
    ax.set_yscale('log')
    ax.set_ylim(YLO, 1.0)
    ax.set_xlim(xlo, -8)
    ax.axhline(THR, ls=':', lw=1.1, color='0.55', zorder=1)
    ax.text(-8.3, THR*1.12, '10% SER ', va='bottom', ha='right', fontsize=8, color='0.45')
    if guess:
        g = 1 - 1.0/n_classes
        ax.axhline(g, ls=':', lw=1.1, color='0.75', zorder=1)
        ax.text(-8.3, g*0.93, f'random guess  {g*100:.1f}% ', va='top', ha='right',
                fontsize=8, color='0.6')
    ax.set_xlabel('SNR (dB, synthetic AWGN on captures)')
    ax.set_ylabel('Symbol error rate')
    ax.set_title(title, fontsize=11)
    ax.grid(which='major', axis='both', alpha=.25, lw=.6)
    ax.grid(which='minor', axis='y', alpha=.12, lw=.5)
    ax.set_axisbelow(True)


def mark_crossings(ax, cross, fs=7.5):
    """10% 선까지 드롭라인 + 값. 그림 하나로 결론이 끝나야 한다.

    교차점이 0.3 dB 까지 붙으므로 라벨을 두 줄로 엇갈려 놓는다.
    """
    for i, (c, col) in enumerate(sorted(cross)):
        ax.plot([c, c], [YLO, THR], ls='-', lw=.9, color=col, alpha=.55, zorder=2)
        ax.plot([c], [THR], marker='v', ms=4.5, color=col, mew=0, zorder=4)
        ax.annotate(f'{c:.2f} dB', (c, YLO * (1.25 if i % 2 else 2.4)),
                    ha='center', va='center', fontsize=fs, color=col, zorder=5,
                    bbox=dict(fc='white', ec='none', alpha=.85, pad=1.2))


def fig1():
    ser = load('ext_sf7_plain.json' if have('ext_sf7_plain.json')
               else 'dnn_sf7_heldout_plain.json')
    short = ()
    if have('ext_sf7_dnn.json'):
        ser['nelora_dnn'] = load('ext_sf7_dnn.json')['nelora_dnn']
    else:
        ser['nelora_dnn'] = load('dnn_sf7_heldout_plain.json')['nelora_dnn']
        short = ('nelora_dnn',)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    xlo, cross = plot_arms(ax, ser, ORDER, short)
    style(ax, 'SF7, held-out packets — their conditions, same batch', xlo)
    mark_crossings(ax, cross)
    ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(.995, .93), framealpha=.95)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig1_ser.png'), dpi=200)
    print('  fig1_ser.png   ' + '  '.join(f'{c:.2f}' for c, _ in cross))


def fig2():
    a = load('ext_sf7_plain.json' if have('ext_sf7_plain.json')
             else 'dnn_sf7_heldout_plain.json')
    b = load('ext_sf7_lpf.json' if have('ext_sf7_lpf.json')
             else 'dnn_sf7_heldout_lpf.json')
    titles = ['Full-band input', 'Band-limited input (+-BW/2)']
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), sharey=True)
    xlo, cr = -8, []
    for ax, ser in zip(axes, (a, b)):
        lo, c = plot_arms(ax, ser, ('decode_loraphy', 'standard_rx'))
        xlo = min(xlo, lo); cr.append(c)
    for ax, t, c in zip(axes, titles, cr):
        style(ax, t, xlo)
        mark_crossings(ax, c)
    axes[1].set_ylabel('')
    axes[0].legend(fontsize=8, loc='upper right', bbox_to_anchor=(.995, .93), framealpha=.95)
    for ax, txt, col in ((axes[0], 'gap  +3.33 dB' + chr(10) + '[+2.67, +4.02]', '#B03030'),
                         (axes[1], 'gap  +0.28 dB' + chr(10) + '[-0.14, +0.70]   n.s.', '#2E7D32')):
        ax.text(.03, .96, txt, transform=ax.transAxes, va='top', ha='left',
                fontsize=9.5, color=col, weight='bold',
                bbox=dict(fc='white', ec=col, lw=.8, alpha=.92, pad=4))
    fig.suptitle('Band-limiting every arm removes the gap — only decode_loraphy moves',
                 fontsize=11, y=.99)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig2_bandlimit.png'), dpi=200)
    print('  fig2_bandlimit.png   ' + ' | '.join(
        '  '.join(f'{c:.2f}' for c, _ in cc) for cc in cr))


def fig3():
    sf = [7, 8, 9, 10]
    g10 = [4.14, 3.52, 3.92, 3.50]
    g20 = [3.38, 3.21, 3.10, 3.03]
    g30 = [3.20, 3.10, 2.95, 2.92]
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    w, x = .26, np.arange(len(sf))
    # 흑백 인쇄: 색 + 해칭을 같이 다르게.
    for i, (g, lb, c, h) in enumerate([(g10, '10% SER', '#2B5FA8', ''),
                                       (g20, '20% SER', '#7BA3D0', '//'),
                                       (g30, '30% SER', '#C9D9EC', 'xx')]):
        ax.bar(x + (i-1)*w, g, w, label=lb, color=c, hatch=h,
               edgecolor='white', lw=.6, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f'SF{s}' for s in sf])
    ax.set_ylabel('decode_loraphy - Standard RX  (dB)')
    ax.set_title('The gap is not specific to SF7' + chr(10) +
                 '(as-run, their filter, nobody aligned)', fontsize=10)
    ax.grid(axis='y', alpha=.25, lw=.6); ax.set_axisbelow(True)
    ax.legend(fontsize=8); ax.set_ylim(0, 4.8)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig3_sf.png'), dpi=200)
    print('  fig3_sf.png')


if __name__ == '__main__':
    print('그림 생성:')
    fig1(); fig2(); fig3()
    print(f'-> {R}')
