# -*- coding: utf-8 -*-
"""
nelora_dnn_eval.py 가 낸 JSON 에서 교차점과 격차를 뽑는다.

micro 와 macro 를 모두 찍는다. 그들 test() 는 코드별 정확도의 macro 평균을 쓰고
(`np.nanmean(div_result, axis=1)`), 이 보고서의 다른 곳은 심볼 micro 평균을 쓴다.
둘 다 보여야 어느 쪽으로 읽어도 결론이 같은지 확인된다.

사용법:
  python nelora_dnn_report.py results/dnn_sf7_asrun.json
"""
import json, argparse
from nelora_mf_test import cross

THR = (10.0, 20.0, 30.0)
NAMES = {'decode_loraphy': 'decode_loraphy (U=100)',
         'nelora_dnn': 'NELoRa DNN',
         'standard_rx': '표준 RX (학습·정렬 없음)'}


def crossings(res, snrs, key, avg):
    ser = [(1 - res[key][f'{s}'][f'{avg}_acc'])*100 for s in snrs]
    return {t: cross(list(snrs), ser, t) for t in THR}


def main(path):
    d = json.load(open(path, encoding='utf-8'))
    snrs, res, fs = d['snrs'], d['results'], d['filter_stat']
    print(f"SF{d['sf']}  평가 {d['n_symbols']}심볼  "
          f"필터 {fs['kept']}/{fs['total']} 통과 ({fs['keep_rate']*100:.2f}%)  "
          f"U={d['upsampling']}")
    print(f"  {d['note']}")

    for avg in ('micro', 'macro'):
        tag = '내 방식' if avg == 'micro' else '그들 test() 방식'
        print(f'\n=== {avg} 평균 ({tag}) — 도달 SNR ===')
        print(f"  {'검출기':<28}" + ''.join(f'{t:>8.0f}%' for t in THR))
        cr = {k: crossings(res, snrs, k, avg) for k in NAMES}
        for k, nm in NAMES.items():
            print(f'  {nm:<28}' + ''.join(
                f'{cr[k][t]:>9.2f}' if cr[k][t] is not None else f'{"없음":>9}' for t in THR))
        b = cr['decode_loraphy']
        print(f'\n  baseline 대비 이득 (+ 가 좋음)')
        for k in ('nelora_dnn', 'standard_rx'):
            print(f'  {NAMES[k]:<28}' + ''.join(
                f'{b[t]-cr[k][t]:>+9.2f}' if (b[t] and cr[k][t]) else f'{"-":>9}' for t in THR))
        a, c = cr['nelora_dnn'][10.0], cr['standard_rx'][10.0]
        if a and c:
            print(f'\n  >> 표준 RX − DNN (10% SER) = {a-c:+.2f} dB')

    print('\n=== SNR 별 정확도 (micro) ===')
    print(f"  {'SNR':>6}{'baseline':>11}{'DNN':>10}{'표준RX':>10}   DNN−baseline")
    for s in snrs:
        bb = res['decode_loraphy'][f'{s}']['micro_acc']
        dd = res['nelora_dnn'][f'{s}']['micro_acc']
        ss = res['standard_rx'][f'{s}']['micro_acc']
        mark = '  <- DNN 이 baseline 보다 나쁨' if dd < bb else ''
        print(f'  {s:>6.0f}{bb*100:>10.1f}%{dd*100:>9.1f}%{ss*100:>9.1f}%'
              f'{(dd-bb)*100:>+13.1f}%p{mark}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('json')
    a = ap.parse_args()
    main(a.json)
