# -*- coding: utf-8 -*-
"""이론 SER 곡선과 SNR 규약 환산.

이 저장소의 모든 SNR 축은 **그들 코드의 규약**이다:

    amp = 10^(-snr/20) * mean|y|          # 샘플당 잡음 진폭
    noise = amp/sqrt(2) * (randn + j*randn)

즉 잡음 전력이 **샘플률 fs = 1 MHz 전대역** 기준으로 정의된다. LoRa 신호는
BW = 125 kHz 안에만 있으므로, 같은 상황을 in-band 기준으로 적으면
10log10(fs/BW) = 10log10(8) = **9.03 dB 높다**. 어느 규약인지 적지 않으면
9 dB 가 조용히 사라진다.

이 모듈이 주는 것:

  (1) 등에너지·직교 M-ary 신호집합의 **비결맞음(noncoherent) 최적 검출** SER 정확식.
      LoRa 심볼은 |cos| = 0.0035 로 사실상 직교이고(본편 README §4.1) 초기 위상이
      미지이므로 이것이 맞는 이론 곡선이다.
  (2) Es/N0 <-> in-band SNR <-> 전대역 SNR 환산.
  (3) 10 / 20 / 30% SER 교차점 표와 곡선의 급경사.

정확식 (Proakis, 비결맞음 직교 M-ary):

    u = |z|^2/N0 로 정규화하면
      오답 가지 M-1 개:  u ~ Exp(1),  CDF = 1 - exp(-u)
      정답 가지:         f(u) = exp(-(u+rho)) * I_0(2*sqrt(rho*u))    (rho = Es/N0)
    P_correct = \\int_0^inf f(u) * (1-exp(-u))^(M-1) du

I_0 가 터지므로 지수 스케일판 ive 로 쓴다:
    f(u) = exp(-(sqrt(u)-sqrt(rho))^2) * ive(0, 2*sqrt(rho*u))

교대급수판 sum (-1)^(n+1) C(M-1,n)/(n+1) exp(-n/(n+1)*rho) 은 M=128 에서 자리수가
전부 날아가므로 쓰지 않는다 — M=2,4,8 에서 적분판 대조용으로만 쓴다.

사용법:
  python theory_ser.py                      # 온전성 검사 + SF7~10 표
  python theory_ser.py --json results/theory_ser.json
"""
import argparse
import json

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import comb, ive


# ------------------------------------------------------------------ 정확식
def ser_noncoherent(rho, M):
    """Es/N0 = rho (선형), 신호 M 개. 비결맞음 직교 검출의 SER."""
    if rho <= 0:
        return 1.0 - 1.0/M
    sr = np.sqrt(rho)

    def integrand(u):
        a = np.sqrt(u) - sr
        return (np.exp(-a*a)*ive(0.0, 2.0*sr*np.sqrt(u))
                * (-np.expm1(-u))**(M - 1))

    # 정답 가지는 u ~= rho 근처에 몰린다. 꼬리를 넉넉히 잡는다.
    hi = rho + 40.0 + 12.0*sr
    pc, _ = quad(integrand, 0.0, hi, limit=400, epsabs=1e-13, epsrel=1e-12)
    return float(min(max(1.0 - pc, 0.0), 1.0 - 1.0/M))


def ser_altsum(rho, M):
    """교대급수 정확식. M 이 작을 때만 유효 (M>=16 에서 자리수 손실)."""
    return float(sum((-1.0)**(n + 1)*comb(M - 1, n)/(n + 1.0)
                     * np.exp(-n/(n + 1.0)*rho) for n in range(1, M)))


def ser_unionbound(rho, M):
    """합집합 상한 (M-1)/2 * exp(-rho/2)."""
    return float(min(1.0, (M - 1)/2.0*np.exp(-rho/2.0)))


# ------------------------------------------------------------------ 규약 환산
class Conv:
    """SNR 규약 환산. OSF = fs/BW (이 데이터셋은 1 MHz / 125 kHz = 8)."""

    def __init__(self, sf, osf=8):
        self.sf, self.osf, self.N = sf, osf, 2**sf

    def esn0_to_inband(self, esn0_db):
        """Es/N0 -> in-band SNR. Es = Ps*N/BW 이고 잡음은 N0*BW 이므로 차이는 N."""
        return esn0_db - 10.0*np.log10(self.N)

    def esn0_to_fullband(self, esn0_db):
        """Es/N0 -> 전대역(fs) SNR. in-band 보다 10log10(osf) 낮다."""
        return self.esn0_to_inband(esn0_db) - 10.0*np.log10(self.osf)

    def fullband_to_esn0(self, fb_db):
        return fb_db + 10.0*np.log10(self.N) + 10.0*np.log10(self.osf)

    @property
    def inband_offset_db(self):
        return 10.0*np.log10(self.osf)


def esn0_at_ser(target, M, lo_db=-10.0, hi_db=40.0):
    """SER = target 이 되는 Es/N0 (dB)."""
    return float(brentq(lambda d: ser_noncoherent(10.0**(d/10.0), M) - target,
                        lo_db, hi_db, xtol=1e-6))


def curve(sf, osf=8, fullband_db=None):
    """전대역 SNR 격자에서 이론 SER. 측정 곡선과 같은 축에 겹칠 수 있다."""
    c = Conv(sf, osf)
    if fullband_db is None:
        fullband_db = np.arange(-40.0, -4.99, 0.25)
    return [(float(fb), ser_noncoherent(10.0**(c.fullband_to_esn0(fb)/10.0), c.N))
            for fb in np.asarray(fullband_db, float)]


# ------------------------------------------------------------------ 온전성 검사
def selftest():
    print('='*78)
    print('T1  M=2 는 닫힌 형태 0.5*exp(-rho/2) 와 같아야 한다')
    ok = True
    for d in (0.0, 5.0, 10.0, 15.0):
        rho = 10.0**(d/10.0)
        a, b = ser_noncoherent(rho, 2), 0.5*np.exp(-rho/2.0)
        print(f'  Es/N0={d:5.1f} dB   적분 {a:.10f}   닫힌형태 {b:.10f}   '
              f'상대오차 {abs(a-b)/b:.2e}')
        ok &= abs(a - b)/b < 1e-9

    print('\nT2  교대급수판과 대조 (M=4, 8 — 아직 자리수가 남는 범위)')
    for M in (4, 8):
        for d in (5.0, 10.0):
            rho = 10.0**(d/10.0)
            a, b = ser_noncoherent(rho, M), ser_altsum(rho, M)
            print(f'  M={M:<3} Es/N0={d:5.1f} dB   적분 {a:.10f}   교대급수 {b:.10f}   '
                  f'상대오차 {abs(a-b)/max(b, 1e-300):.2e}')
            ok &= abs(a - b)/max(b, 1e-300) < 1e-8

    print('\nT3  저SNR 극한 — rho -> 0 에서 1 - 1/M 로 가야 한다')
    for M in (2, 128):
        v = ser_noncoherent(10.0**(-40.0/10.0), M)
        print(f'  M={M:<4} SER(Es/N0=-40dB) = {v:.6f}   (1-1/M = {1-1/M:.6f})')
        ok &= abs(v - (1 - 1.0/M)) < 2e-3

    print('\nT4  합집합 상한이 정확식 위에 있어야 한다 (M=128)')
    for d in (8.0, 10.0, 12.0, 14.0):
        rho = 10.0**(d/10.0)
        a, u = ser_noncoherent(rho, 128), ser_unionbound(rho, 128)
        print(f'  Es/N0={d:5.1f} dB   정확 {a:.6f}   상한 {u:.6f}   '
              + ('OK' if u >= a else '** 위반'))
        ok &= u >= a

    print('\n  ' + ('OK — 정확식을 믿고 쓴다.' if ok else '** 실패: 여기서 멈출 것.'))
    return ok


# ------------------------------------------------------------------ 표
def table(sfs=(7, 8, 9, 10), osf=8, targets=(0.10, 0.20, 0.30)):
    rows = []
    for sf in sfs:
        c = Conv(sf, osf)
        rec = dict(sf=sf, M=c.N, osf=osf, inband_offset_db=c.inband_offset_db)
        for t in targets:
            e = esn0_at_ser(t, c.N)
            rec[f'ser{int(t*100)}'] = dict(
                esn0_db=e, inband_db=c.esn0_to_inband(e),
                fullband_db=c.esn0_to_fullband(e))
        rows.append(rec)

    print('='*78)
    print(f'이론 교차점 — 비결맞음 직교 M-ary 최적 검출, OSF = fs/BW = {osf}')
    print(f'  전대역 SNR = in-band SNR - {10*np.log10(osf):.2f} dB'
          '   (그들 코드가 쓰는 축이 전대역이다)')
    for t in targets:
        k = f'ser{int(t*100)}'
        print(f'\n  --- {int(t*100)}% SER ---')
        print(f"  {'SF':>3}{'M':>6}{'Es/N0':>11}{'in-band SNR':>15}{'전대역 SNR':>15}")
        for r in rows:
            d = r[k]
            print(f"  {r['sf']:>3}{r['M']:>6}{d['esn0_db']:>9.2f} dB"
                  f"{d['inband_db']:>12.2f} dB{d['fullband_db']:>12.2f} dB")
    return rows


def slope_table(sf=7, osf=8):
    """'기울기가 완만하다' 를 숫자로 적기 위한 것 — 이론 곡선의 급경사 폭."""
    c = Conv(sf, osf)
    e90, e10, e01 = (esn0_at_ser(t, c.N) for t in (0.90, 0.10, 0.01))
    print('\n' + '='*78)
    print(f'SF{sf} 이론 곡선의 급경사 — 전대역 축에서')
    print(f'  SER 90%  {c.esn0_to_fullband(e90):+7.2f} dB')
    print(f'  SER 10%  {c.esn0_to_fullband(e10):+7.2f} dB')
    print(f'  SER  1%  {c.esn0_to_fullband(e01):+7.2f} dB')
    print(f'  -> 90%->10% 폭 {e10-e90:4.2f} dB,  10%->1% 폭 {e01-e10:4.2f} dB')
    return dict(ser90_fullband_db=c.esn0_to_fullband(e90),
                ser10_fullband_db=c.esn0_to_fullband(e10),
                ser01_fullband_db=c.esn0_to_fullband(e01),
                width_90_to_10_db=e10 - e90, width_10_to_1_db=e01 - e10)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--osf', type=int, default=8)
    ap.add_argument('--sfs', default='7,8,9,10')
    ap.add_argument('--json', default=None)
    ap.add_argument('--no-selftest', action='store_true')
    a = ap.parse_args()
    if not a.no_selftest and not selftest():
        raise SystemExit(1)
    print()
    sfs = [int(v) for v in a.sfs.split(',')]
    rows = table(sfs, a.osf)
    sl = slope_table(sfs[0], a.osf)
    if a.json:
        json.dump(dict(osf=a.osf, crossings=rows, slope_sf7=sl,
                       curves={str(sf): curve(sf, a.osf) for sf in sfs}),
                  open(a.json, 'w'), indent=2)
        print(f'\n저장: {a.json}')
