# -*- coding: utf-8 -*-
"""
5.8% 바닥의 정체를 캔다.

질문:
  Q1  세 검출기가 '같은' 심볼에서 틀리는가?
        겹친다 -> 데이터/라벨 문제.  안 겹친다 -> 검출기 문제.
  Q2  틀리는 심볼이 특정 패킷/코드/심볼위치에 몰려 있는가?
  Q3  틀린 심볼은 원래 약한(저SNR) 캡처인가?
  Q4  code_label 이 재래식 디코더(=D1) 출력으로 만들어진 것은 아닌가?
        그렇다면 D1 은 '정답을 만든 알고리즘' 이므로 비교 자체가 기울어 있다.
  Q5  파일명 4번째 필드는 무엇인가?

사용법:
  python nelora_floor.py <데이터폴더> [--sf 7] [--n 6000]
"""
import os, glob, argparse
from collections import Counter
import numpy as np

from nelora_mf_test import (ideal_chirps, norm_rows, d1_nelora, d_mf,
                            empirical_prototypes, parse_raw_name, load_raw)


def est_snr(Y, down, osf):
    """dechirp 후 신호가 접힌 두 빈에 실린 에너지 vs 나머지 = 캡처별 SNR 추정."""
    M = Y.shape[1]
    Z = np.fft.fft(Y*down[None, :], M, axis=1)/np.sqrt(M)
    P = np.abs(Z)**2
    T = int(round(M/osf))
    fold = P[:, :T] + P[:, -T:]                 # 두 alias 를 합친 빈별 에너지
    k = np.argmax(fold, 1)
    Es = fold[np.arange(len(Y)), k]
    En = P.sum(1) - Es
    return 10*np.log10(Es/np.maximum(En, 1e-30))


def main(root, sf_want, n_max, proto_pkts, max_clean):
    files = glob.glob(os.path.join(root, '**', '*.mat'), recursive=True)
    recs = [m for m in (parse_raw_name(f) for f in sorted(files)) if m and m['sf'] == sf_want]
    if not recs:
        print('해당 SF 파일 없음'); return

    Pid, osf, M, N = ideal_chirps(sf_want, 125e3)
    down = np.conj(Pid[0]); Pidn = norm_rows(Pid)

    # ---------------------------------------------------------------- Q5
    print('='*70)
    print('Q5  파일명 4번째 필드')
    v4 = Counter()
    for f in files:
        p = os.path.splitext(os.path.basename(f))[0].split('_')
        if len(p) == 4:
            v4[p[2]] += 1
    print(f'  전체 SF 통틀어 값 분포: {dict(v4)}')
    print('  -> 단일값이면 이 릴리스에서는 정보가 없는 필드 (배치/버전 자리로 추정)')
    sym_per_pkt = Counter(r['packet'] for r in recs)
    print(f'  심볼 인덱스 범위: {min(r["symbol"] for r in recs)} ~ {max(r["symbol"] for r in recs)}')
    print(f'  패킷당 심볼 수: 최소 {min(sym_per_pkt.values())}, 최대 {max(sym_per_pkt.values())}')

    # preamble(0~7) 의 code_label 이 전부 0 인지 = 라벨이 프로토콜 구조와 맞는지
    pre = [r for r in recs if r['symbol'] <= 7]
    print(f'  symbol_idx 0~7 의 code_label 이 0 인 비율: '
          f'{np.mean([r["code_label"] == 0 for r in pre])*100:.1f}%  (n={len(pre)})')

    # ---------------------------------------------------------------- 적재
    pkts = sorted({r['packet'] for r in recs}, key=int)
    proto_set = set(pkts[:proto_pkts])
    proto = [r for r in recs if r['packet'] in proto_set and r['symbol'] >= 12]
    test  = [r for r in recs if r['packet'] not in proto_set and r['symbol'] >= 12]

    by = {c: [] for c in range(N)}
    for r in proto:
        c = r['code_label'] % N
        if len(by[c]) < max_clean:
            y = load_raw(r['path'])
            if y.size == M:
                by[c].append(y)
    Pemp, frac, cnt = empirical_prototypes(by, N, M)

    rng = np.random.default_rng(0)
    if len(test) > n_max:
        test = [test[i] for i in rng.choice(len(test), n_max, replace=False)]
    Y, lab, pkt, sidx = [], [], [], []
    for r in test:
        y = load_raw(r['path'])
        if y.size == M:
            Y.append(y); lab.append(r['code_label'] % N)
            pkt.append(r['packet']); sidx.append(r['symbol'])
    Y = np.array(Y); lab = np.array(lab)
    pkt = np.array(pkt); sidx = np.array(sidx)

    p1 = d1_nelora(Y, down, osf)
    p2 = d_mf(Y, Pidn)
    p3 = d_mf(Y, Pemp)
    e1, e2, e3 = p1 != lab, p2 != lab, p3 != lab
    print(f'\n평가 심볼 {len(Y)}개 (합성잡음 없음, 캡처 원본)')
    print(f'  D1 {e1.mean()*100:.2f}%   D2 {e2.mean()*100:.2f}%   D3 {e3.mean()*100:.2f}%')

    # ---------------------------------------------------------------- Q1
    print('\n' + '='*70)
    print('Q1  세 검출기가 같은 심볼에서 틀리는가')
    allw = e1 & e2 & e3
    anyw = e1 | e2 | e3
    print(f'  하나라도 틀린 심볼 : {anyw.sum():5d} ({anyw.mean()*100:.2f}%)')
    print(f'  셋 다 틀린 심볼    : {allw.sum():5d} ({allw.mean()*100:.2f}%)')
    if anyw.sum():
        print(f'  -> 틀린 심볼 중 셋 다 틀린 비율 = {allw.sum()/anyw.sum()*100:.1f}%')
    for nm, m in (('D1만', e1 & ~e2 & ~e3), ('D2만', e2 & ~e1 & ~e3), ('D3만', e3 & ~e1 & ~e2),
                  ('D1+D2', e1 & e2 & ~e3), ('D1+D3', e1 & e3 & ~e2), ('D2+D3', e2 & e3 & ~e1)):
        print(f'    {nm:6s} 만 틀림: {m.sum():4d}')
    # 셋 다 틀릴 때 예측끼리도 일치하는가 = 신호가 실제로 그 코드로 보인다는 뜻
    if allw.sum():
        same = ((p1 == p2) & (p2 == p3) & allw).sum()
        print(f'  셋 다 틀린 심볼 중 세 예측이 서로 일치: {same}/{allw.sum()} '
              f'({same/allw.sum()*100:.1f}%)')
        print('  -> 높으면 라벨이 틀렸거나 캡처가 실제로 다른 코드를 담고 있다는 뜻')

    # ---------------------------------------------------------------- Q2
    print('\n' + '='*70)
    print('Q2  오류가 특정 패킷/코드/심볼위치에 몰려 있는가')
    for nm, key in (('패킷', pkt), ('코드', lab), ('심볼위치', sidx)):
        tot, bad = Counter(), Counter()
        for k, b in zip(key, allw):
            tot[k] += 1; bad[k] += bool(b)
        rates = np.array([bad[k]/tot[k] for k in tot if tot[k] >= 5])
        share = sorted(((bad[k], k, tot[k]) for k in tot), reverse=True)[:5]
        nz = (rates > 0).sum()
        print(f'  [{nm}] 그룹 {len(tot)}개 중 오류가 하나라도 있는 그룹 {nz}개 '
              f'({nz/max(len(rates),1)*100:.0f}%), 그룹별 오류율 중앙값 {np.median(rates)*100:.1f}% '
              f'/ 최대 {rates.max()*100:.1f}%')
        print(f'        오류 최다: ' + ', '.join(f'{k}({b}/{t})' for b, k, t in share))
    # 상위 몇 개 패킷이 전체 오류의 몇 %를 먹는가
    tot, bad = Counter(), Counter()
    for k, b in zip(pkt, allw):
        tot[k] += 1; bad[k] += bool(b)
    order = sorted(bad.values(), reverse=True)
    if sum(order):
        top = max(1, len(order)//10)
        print(f'  -> 오류 최다 상위 10% 패킷({top}개)이 전체 오류의 '
              f'{sum(order[:top])/sum(order)*100:.1f}% 를 차지')

    # ---------------------------------------------------------------- Q3
    print('\n' + '='*70)
    print('Q3  틀린 심볼은 원래 약한 캡처인가')
    snr = est_snr(Y, down, osf)
    print(f'  캡처별 추정 SNR: 중앙값 {np.median(snr):.1f} dB, '
          f'10퍼센타일 {np.percentile(snr,10):.1f} dB, 90퍼센타일 {np.percentile(snr,90):.1f} dB')
    print(f'  맞힌 심볼 평균 {snr[~allw].mean():.2f} dB   틀린 심볼 평균 {snr[allw].mean():.2f} dB')
    q = np.percentile(snr, [20, 40, 60, 80])
    binid = np.digitize(snr, q)
    print('  추정SNR 5분위별 오류율:')
    for b in range(5):
        m = binid == b
        if m.sum():
            print(f'    Q{b+1} (SNR {snr[m].min():6.1f}~{snr[m].max():6.1f} dB, n={m.sum():4d}): '
                  f'D1 {e1[m].mean()*100:5.1f}%  D2 {e2[m].mean()*100:5.1f}%  D3 {e3[m].mean()*100:5.1f}%')

    # ---------------------------------------------------------------- Q4
    print('\n' + '='*70)
    print('Q4  라벨이 D1(재래식 디코더) 출력으로 만들어졌을 가능성')
    dis = p1 != p2
    if dis.sum():
        w1 = (lab[dis] == p1[dis]).mean()*100
        w2 = (lab[dis] == p2[dis]).mean()*100
        print(f'  D1 과 D2 의 예측이 갈린 심볼 {dis.sum()}개 중')
        print(f'    라벨이 D1 쪽 = {w1:.1f}%      라벨이 D2 쪽 = {w2:.1f}%      둘 다 아님 = {100-w1-w2:.1f}%')
    dis13 = p1 != p3
    if dis13.sum():
        w1 = (lab[dis13] == p1[dis13]).mean()*100
        w3 = (lab[dis13] == p3[dis13]).mean()*100
        print(f'  D1 과 D3 의 예측이 갈린 심볼 {dis13.sum()}개 중')
        print(f'    라벨이 D1 쪽 = {w1:.1f}%      라벨이 D3 쪽 = {w3:.1f}%      둘 다 아님 = {100-w1-w3:.1f}%')
    print('  -> D1 쪽이 압도적이면 라벨 생성기가 D1 계열이라는 뜻이고,')
    print('     그러면 D1 의 SER 은 낮게 나올 수밖에 없다 (자기가 만든 정답을 맞히는 셈).')

    # 오류의 모양: 인접 빈 오류인가 무작위인가
    print('\n  오류 형태 (예측 - 라벨, mod N):')
    for nm, p, e in (('D1', p1, e1), ('D2', p2, e2), ('D3', p3, e3)):
        d = (p[e] - lab[e]) % N
        d = np.where(d > N//2, d - N, d)
        near = (np.abs(d) <= 1).mean()*100 if len(d) else 0
        print(f'    {nm}: |차이|<=1 인 비율 {near:5.1f}%   중앙 |차이| {np.median(np.abs(d)) if len(d) else 0:.0f} bin')
    print('  -> 인접 빈 오류가 많으면 타이밍/CFO 문제, 무작위면 잡음/라벨 문제')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--sf', type=int, default=7)
    ap.add_argument('--n', type=int, default=6000)
    ap.add_argument('--proto-pkts', type=int, default=60)
    ap.add_argument('--max-clean', type=int, default=30)
    a = ap.parse_args()
    main(a.root, a.sf, a.n, a.proto_pkts, a.max_clean)
