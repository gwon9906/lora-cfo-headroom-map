# LoRa-BAM Reconstruction

> **초저 SNR LoRa 복조 성공률을 올리기 위해 Complex-valued BAM 기반 신호 복원을 시도했고, 재검증 끝에 실제 개선 요인이 검출기에 있음을 찾아 -25 dB 성공률을 12% → 54% 로 끌어올린 연구. 동시에 1차 결과와 1차 결론을 스스로 반증하고, 학습이 의미를 갖는 조건(시변 CFO · sequence-level)을 상한과 함께 정량화했다.**
> 2025.09 – 2026.02 · 개인 연구 / 교수 지도
>
> ⚠️ §3.2 / §4.1 은 이 저장소의 1차 결론을 **스스로 반증한 재검증 결과**다. 1차 결과표와 SOM 기반 종료 근거는 더 이상 유효하지 않으며, 그 이유와 대체 근거를 §4–§6 에 정리했다.

---

## 1. 문제 정의

- LoRa는 CSS(Chirp Spread Spectrum) 기반 변조로, de-chirp + FFT argmax로 심볼을 복조한다.
- **초저 SNR(-25 ~ -30 dB) 구간에서는 이 기존 복조 파이프라인이 무너진다.**
- 본 연구의 **당초** 목표는 **신경망 기반 신호 복원(denoising)** 을 앞단에 붙여 복조 가능한 SNR 하한을 낮출 수 있는지 검증하는 것이었다.
- 재검증 결과 이 설정은 AWGN 하에서 **원리적으로 닫혀 있었고**(§4.2), 질문은 다음으로 교체되었다:
  > *dechirp 통계량이 최적이 아니게 되는 조건은 무엇이며, 그 조건에서 남는 여지는 정량적으로 얼마인가?*

## 2. 접근

- **Complex-valued BAM (Bidirectional Associative Memory) 설계**
  - 가중치 `W_uv`, `W_vu` 를 실제 복소수로 유지 (실수 concat 아님).
  - 활성함수는 **split-tanh** — IEEE 논문의 Lagrange 안정성 조건을 만족하도록 선택.
  - 동역학은 Euler 적분으로 forward-backward를 반복.
- **구조 실험**
  - Noise2Noise 자가지도 학습
  - 스킵 커넥션 / outer residual / tied-untied encoder 등 다수 변형
  - Complex 스펙트로그램 입력 · IQ 직접 입력 두 도메인 모두 시도

## 3. 결과 — 1차 결과와, 그것을 무효화한 재검증

### 3.1 1차 결과 (BAMv3 + Noise2Noise)

일부 초저 SNR 구간에서 repo 내 baseline 대비 개선이 관측되었다.

![결과 요약표](local/bestresult.png)

- -30 dB: 3.5% → **7.1%** (+3.6%p)
- -25 dB: 18.7% → **27.8%** (+9.1%p)

![복조 성능 곡선](local/bestresultGraph.png)

**복원 예시 (SNR = -15 dB, Complex spectrogram 도메인)**

![복원 스펙트로그램](local/denoised.png)
좌: 노이즈 입력 · 중: BAM 복원 (bottleneck=512) · 우: 클린 타깃.
Chirp 궤적은 일부 살아나지만 진폭 collapse가 남아 있다.

### 3.2 재검증 — 이 개선은 baseline의 구현 손실을 되찾은 것이었다

`verification/final.py` 에서 **동일한 노이즈 실현**으로 세 detector를 비교했다 (3000 trials/SNR).

| SNR | README 부록 이전 버전 `power[:512]` | repo detector (OSF-fold) | **matched filter (= ML)** |
|---|---|---|---|
| -30 dB | 2.3% | 1.8% | **8.7%** |
| -25 dB | 12.2% | 12.3% | **51.9%** |
| -20 dB | 42.9% | 72.3% | **99.5%** |
| -15 dB | 65.0% | 99.0% | **100.0%** |

`verification/why_gap.py` 로 원인을 분리했다. dechirp 후 2048-point FFT에서 clean 심볼의 에너지가
실제로 어디에 놓이는지 보면:

```
sym 100: top FFT bins [99, 100, 1636, 1637]   상위 4개 bin 에너지 = 75.3%   blocks = [0, 3]
sym 300: top FFT bins [299, 300, 1836, 1837]  상위 4개 bin 에너지 = 70.4%   blocks = [0, 3]
```

**이 파이프라인의 dechirp+FFT는 심볼 에너지를 하나의 bin에 모으지 못한다.**
에너지가 인접 bin으로 새고(on-grid가 아님), 동시에 멀리 떨어진 두 alias 블록으로 쪼개진다.
argmax는 그 중 한 조각만 본다. 여러 folding 변형을 같은 노이즈로 비교하면:

| detector | -30 dB | -25 dB | -20 dB | -15 dB |
|---|---|---|---|---|
| pre-fix `power[:512]` | 2.2% | 13.1% | 43.6% | 63.1% |
| repo OSF-fold (4 블록) | 1.6% | 12.2% | 74.2% | 99.1% |
| 신호 블록 2개만 non-coherent 합 | 1.4% | 9.6% | 38.2% | 61.8% |
| 신호 블록 2개 coherent 합 | 0.7% | 4.5% | 23.4% | 53.6% |
| **matched filter (ML)** | **8.1%** | **52.8%** | **99.6%** | **100.0%** |

folding 방식을 어떻게 바꿔도 MF와의 격차는 메워지지 않는다. 즉 이것은 folding 개수 문제가 아니라
**이 dechirp+FFT 구현이 충분통계량이 아니라는 문제**다. 프로토타입과의 상관(MF)만이 흩어진 에너지를
전부 coherent하게 모은다. 격차는 SNR 환산 약 **5 dB** 다 (repo -20 dB ≈ MF -25 dB).

그 결과:

> **BAMv3의 -25 dB 27.8% 는, 같은 조건에서 아무것도 학습하지 않은 matched filter 의 51.9% 에 크게 못 미친다.**
> 즉 1차 결과의 "개선"은 최적 detector를 넘어선 것이 아니라, **baseline이 스스로 버리고 있던 손실의 일부를 신경망이 되찾아준 것**이다.
> 올바른 baseline을 쓰면 이 개선폭은 남지 않는다.

### 3.3 그리고 그 최적 detector는 엣지에서 돌아간다 (연산량 1.5배)

"MF가 최적이지만 512×2048 상관은 기기에서 못 돌린다"는 반론이 자연스럽다. 성립하지 않는다.

`X[m] = roll(X[0], -OSF·m)` 이 **정확히** 성립한다(모든 m에 대해 검증). 즉 512개 프로토타입 상관은
전부 **하나의 순환상관**이고, `IFFT(FFT(y)·conj(FFT(X[0])))` 에서 512개 lag만 읽으면 끝이다.
FFT 2회면 MF 뱅크 전체가 나온다.

| detector | -30 dB | -25 dB | -20 dB | -15 dB | 심볼당 비용 |
|---|---|---|---|---|---|
| repo dechirp+FFT (FFT 1회) | 1.4% | 12.0% | 73.3% | 99.0% | 27.4 µs |
| **FFT-MF (최적, FFT 2회)** | **9.0%** | **54.0%** | **99.6%** | **100.0%** | **41.4 µs** |
| direct MF bank (512×2048) | 9.0% | 54.0% | 99.6% | 100.0% | 2830 µs |

FFT-MF 는 direct MF 와 **판정이 100% 일치**한다(근사가 아니라 등가). 비용은 기존 대비 **1.5배**다.

> **복조 성공률을 올리는 가장 큰 단일 요인은 신경망이 아니라 detector 교체였다.**
> -25 dB 에서 **12.0% → 54.0% (+42%p)**, 학습 없이, 클라우드 없이, FFT 한 번 더 돌려서.
> 같은 지점에서 BAMv3가 만든 개선은 +9%p 였고, 그것은 이 42%p 안에 포함되는 부분집합이었다.

이 사실은 백홀/C-RAN 논의 자체를 무효화한다. **IQ를 클라우드로 보낼 이유가 없다.**
초저 SNR에서 복조가 안 되던 원인의 대부분은 SNR이 아니라 게이트웨이 측 detector 손실이었다.

## 4. 종료 근거 — 두 번 바뀌었다

### 4.1 처음 세웠던 근거 (SOM 군집 실패) — **철회**

처음에는 "초저 SNR LoRa IQ는 심볼 구분 정보를 담고 있지 않다"는 결론을 SOM 실험에서 끌어냈다.

![STFT-SOM Label Map](local/SOM.png)

SOM 격자 위에서 같은 label이 연속 영역을 이루지 못하고 파편화된 것은 **사실**이다.
그러나 `verification/verify.py` 로 **노이즈가 전혀 없는 clean 심볼 512개의 기하 구조**를 직접 재보았다.

```
|cos| between DIFFERENT symbols : mean=0.00350  max=0.02169
normalized L2 distance          : mean=1.41426  min=1.39892  max=1.42947
  -> ratio max/min = 1.0218      (1.0 == 완전 등거리)
mean dist to NEIGHBOUR (m, m+1)  = 1.41283
mean dist to OPPOSITE (m, m+256) = 1.41421
```

**LoRa 심볼 집합은 설계상 서로 거의 직교하며, 따라서 IQ 공간에서 거의 완전한 등거리 배치다.**
심볼 인덱스가 가깝든 멀든 거리는 동일하다 (1.41283 vs 1.41421).

즉 SOM은 **SNR = +∞ 에서도 군집을 만들 수 없다.** 군집 실패는 노이즈나 정보량과 무관하며,
직교 codebook이면 당연히 나오는 결과다. 관측은 맞았지만 **거기서 끌어낸 결론이 틀렸다.**
이 근거는 종료 사유로 쓸 수 없다.

### 4.2 실제 종료 근거 — AWGN 구간에서는 정리(theorem)로 닫힌다

LoRa 심볼 512개는 **등에너지 · 근사 직교 신호 집합**이다 (§4.1의 |cos| ≈ 0.0035 가 그 측정치다).
AWGN 하에서 이런 집합의 최적 검출기는 512개 프로토타입 전체에 대한 noncoherent matched filter이고,
그 상관 벡터 `{|p_m^H y|}` 가 **충분통계량(sufficient statistic)** 이다.
(§3.3 에서 보았듯 이 통계량은 FFT 2회로 정확히 계산된다. 이 저장소가 쓰던 dechirp+FFT 는
그 MF의 손실 있는 변형이었을 뿐, 최적성과 저연산은 양립한다.)

따라서 심볼 단위 front-end `g(y)` 를 앞에 붙이는 모든 구조는 data processing inequality에 의해
**정보를 늘릴 수 없고, 잘해야 보존한다.**

> **AWGN · 심볼 단위 · denoising front-end 라는 조합은 성능 향상 여지가 0 임이 증명되어 있다.**
> BAM이든 CNN이든 diffusion이든 구조를 바꿔도 결과는 같다.
> 이 연구 라인의 종료는 실험적 실패가 아니라 **문제 설정이 닫혀 있었다는 사실**에 근거한다.

§3.2 의 측정은 이 정리의 따름정리(corollary)를 그대로 보여준다 — 개선처럼 보인 것의 정체는
baseline이 최적이 아니었다는 것뿐이었다.

## 5. 그렇다면 열려 있는 조건은 무엇인가 — CFO 재정의

"AWGN에서 닫혀 있다"는 것은 **비최적성이 존재하는 조건에서만 학습 기반 접근이 의미를 가진다**는 뜻이다.
CFO(carrier frequency offset)는 그 조건에 해당한다. CFO가 있으면 zero-CFO 프로토타입에 맞춰진
dechirp 통계량은 더 이상 충분통계량이 아니다. 초저 SNR + 하드웨어 임페어먼트 하에서
신경망 복조를 다루는 기존 연구 라인이 겨냥하는 지점도 여기다.

`verification/cfo_headroom.py`, `verification/pkt_test.py`, `verification/break_test.py` 로
**여지가 실제로 있는지, 있다면 얼마이며, 고전적 방법이 이미 그것을 다 먹는지**를 측정했다.

### 5.1 CFO는 실제로 여지를 만든다

residual CFO ε (bin 단위, 1 bin = 488.28 Hz) 를 고정했을 때 matched filter 정확도:

| SNR | ε=0 | ε=0.10 | ε=0.25 | ε=0.50 |
|---|---|---|---|---|
| -30 dB | 6.7% | 7.4% | 5.9% | 2.2% |
| -25 dB | 55.0% | 49.2% | 39.2% | 10.7% |
| -20 dB | 99.5% | 99.7% | 97.8% | 44.3% |
| -15 dB | 100.0% | 100.0% | 100.0% | 49.4% |

ε=0.5 bin 에서 고 SNR에서도 ≈50% 로 주저앉는 것은 peak이 두 bin 정확히 사이에 놓이는
구조적 모호성이다. **CFO는 SNR로 극복되지 않는 손실을 만든다.**

### 5.2 그러나 여지의 지렛대는 "심볼 단위 denoising"이 아니다

ε ~ U(-0.5, 0.5) bin, 패킷 내 상수 (물리적으로 맞는 모델):

| SNR | blind MF | 심볼단위 joint (sym,CFO) 탐색 | 패킷 ns=8 | ns=32 | ns=128 | genie (CFO 기지) | headroom |
|---|---|---|---|---|---|---|---|
| -30 dB | 6.6% | 7.0% | 7.0% | 6.2% | 6.5% | 9.7% | 3.1%p |
| -25 dB | 33.8% | 38.4% | 41.7% | 46.9% | **53.6%** | 53.6% | 19.8%p |
| -20 dB | 90.0% | 91.6% | 98.4% | 99.7% | **99.7%** | 99.7% | 9.7%p |
| -15 dB | 94.5% | 89.9% | 96.2% | 100.0% | **100.0%** | 100.0% | 5.5%p |

-25 dB 기준 headroom 은 19.8%p 로 작지 않다. 그런데 **심볼 하나만 보는 joint 탐색은
19.8%p 중 4.6%p 밖에 못 가져온다.** 초저 SNR에서 심볼 하나는 자기 CFO를 추정할 SNR 자체가 없다.
반면 심볼을 가로질러 pooling 하면 headroom이 전부 회수된다 (ns=128 → genie와 동일).

> **이것이 06번 실험(심볼 단위 Complex IQ BAM)이 구조적으로 도달할 수 없었던 이유다.**
> CFO 구간의 여지는 심볼 내부가 아니라 **심볼들 사이의 공유 구조**에 있다.
> 어떤 per-symbol autoencoder/BAM도 이 여지에 접근할 수 없다.

### 5.3 그리고 정적 CFO라면 고전적 방법이 이미 상한에 닿는다

패킷 pooling 이 깊은 SNR에서도 genie 상한에 도달하는지 확인:

| SNR | blind | ns=32 | ns=128 | ns=512 | genie | genie − best |
|---|---|---|---|---|---|---|
| -32 dB | 2.6% | 2.0% | 1.6% | 0.6% | 4.8% | 2.8%p |
| -30 dB | 5.4% | 5.5% | 6.4% | 7.9% | 8.2% | 0.3%p |
| -28 dB | 11.4% | 11.3% | 11.1% | 17.8% | 17.4% | -0.4%p |
| -26 dB | 13.5% | 30.9% | 32.4% | 37.4% | 38.0% | 0.6%p |

-28 dB 까지, 단순 grid search 기반 패킷 CFO 추정이 genie 상한과 사실상 같다.
따라서 **"초저 SNR + 정적 CFO 에서 신경망이 고전 방식을 이긴다"는 주장은 성립하지 않는다.**
여기서 신경망의 기여는 정확도가 아니라 연산량뿐이다.

### 5.4 실제로 열려 있는 조건 — CFO가 시간에 따라 변할 때

고전 추정기는 "CFO는 패킷 내 상수"라는 모델에 의존한다. SFO/드리프트로 이 가정이 깨지면:

**SNR = -25 dB, ns=128, 패킷 동안 CFO가 선형으로 drift**

| 패킷 내 drift (bin) | blind MF | 패킷 상수-CFO 추정기 | genie |
|---|---|---|---|
| 0.00 | 42.0% | 53.0% | 53.2% |
| 0.10 | 41.9% | 51.7% | 52.1% |
| 0.25 | 36.5% | 54.1% | 55.2% |
| 0.50 | 35.4% | 43.6% | 49.6% |
| **1.00** | 28.1% | **24.1%** | **52.0%** |

drift = 1 bin 에서 **상수-CFO 추정기는 아무것도 안 한 blind MF보다 오히려 나쁘다 (24.1% < 28.1%).**
모델 불일치가 추정을 능동적으로 망가뜨린다. 반면 genie 상한은 52.0% 그대로다.

> **약 28%p 의 여지가, 어떤 상수-CFO 고전 추정기로도 접근 불가능한 상태로 남아 있다.**
> 이것이 이 연구에서 유일하게 방어 가능한 "열린 조건"이다:
> **초저 SNR + 시변 CFO(drift/SFO) 하의 sequence-level 복조.**
> 상한(genie)이 계산 가능하므로 성과를 수치로 검증할 수 있고,
> 고전 baseline(상수-CFO 패킷 추정기)이 명확히 실패하므로 비교 대상도 분명하다.

## 6. 결론

연구 목표는 **초저 SNR에서 LoRa 복조 성공률을 올리는 것**이었고, 제약은 **게이트웨이/기기에서 돌아가야 한다**는 것이었다(IQ를 클라우드로 올리는 백홀 비용이 이득보다 크다). 그 목표 기준으로 정리하면:

**달성한 것 — -25 dB 복조 성공률 12.0% → 54.0% (+42%p), 엣지에서 동작**

경로는 예상과 달랐다. 개선을 만든 것은 신경망이 아니라, 최적 검출기가 **FFT 2회로 정확히 계산된다**는 사실이었다 (§3.3). 기존 대비 연산량 1.5배, 학습 없음, 클라우드 없음. 백홀 문제는 애초에 풀 필요가 없었다 — 복조가 안 되던 원인의 대부분이 SNR이 아니라 게이트웨이 측 detector 손실이었기 때문이다.

**그 과정에서 스스로 폐기한 것**

1. **1차 결과(BAMv3 +9%p)를 폐기했다.** 그 9%p는 detector 교체가 주는 42%p의 부분집합이었다. -25 dB 에서 BAMv3 27.8% vs 학습 없는 FFT-MF 54.0%. 올바른 baseline 앞에서 남지 않는다. (§3.2)
2. **종료 근거로 쓰던 SOM 분석을 철회했다.** LoRa 심볼은 설계상 등거리라(거리비 1.0218) SOM은 **노이즈가 없어도** 군집을 만들 수 없다. 관측은 옳았고 해석이 틀렸다. 초저 SNR과 무관한 현상을 초저 SNR의 증거로 읽었다. (§4.1)

**대신 얻은 종료 근거 — 문제 설정이 닫혀 있었다**

AWGN에서 프로토타입 상관이 충분통계량이므로, 심볼 단위 front-end는 data processing inequality에 의해 정보를 늘릴 수 없다. BAM이든 CNN이든 diffusion이든 같다. 이 라인의 종료는 실험 실패가 아니라 **정리(theorem)** 에 근거한다. (§4.2)

**남겨둔 것 — 유일하게 열린 조건**

학습이 의미를 갖는 곳은 MF가 최적이 아니게 되는 조건뿐이다. CFO가 그렇다. 다만 범위가 좁다:

- CFO는 -25 dB에서 19.8%p의 여지를 만든다 (blind 33.8% vs genie 53.6%). (§5.1)
- 그러나 그 여지는 **심볼 단위로 도달 불가능하다** — 심볼단위 탐색은 4.6%p만 회수, 패킷 pooling은 전부 회수. 06번 실험이 구조적으로 닿을 수 없었던 이유다. (§5.2)
- **정적 CFO라면 고전 grid search가 이미 genie 상한에 닿는다.** 신경망 우위 주장 불가. (§5.3)
- **시변 CFO(drift/SFO)에서만** 고전 상수-CFO 추정기가 blind보다도 나빠지고(24.1% < 28.1%), **약 28%p가 상한과 함께 미회수로 남는다.** (§5.4)

→ 후속 연구의 유일한 방어 가능한 설정: **초저 SNR + 시변 CFO 하의 sequence-level 복조.** 상한이 계산 가능하고 고전 baseline이 명확히 실패하므로, 성과를 수치로 증명할 수 있다.

**이 저장소의 산출물**

"초저 SNR LoRa에 신경망을 붙여봤다"가 아니라,

> **복조 성공률을 실제로 올린 요인을 찾아내고(+42%p, 엣지 동작), 자신의 1차 결과와 1차 결론을 스스로 반증했으며, 어느 조건이 정리로 닫혀 있고 어느 조건이 상한과 함께 정량화된 여지를 남기는지 수치로 분리해 낸 것.**

### 재현

```
verification/verify.py         # clean 심볼의 등거리성 (§4.1)
verification/detect_test.py    # 초기 탐색용. 여기서 쓴 resample 기반 'ML' 은
                               #   dechirp 톤이 ±125 kHz 밖에도 놓이는 것을 놓쳐
                               #   과소평가된 변형이다. §3.2 수치는 final.py 기준.
verification/final.py          # 동일 노이즈 실현 3-detector 비교 (§3.2)
verification/som_test.py       # SOM purity / contiguity 통계 검정
verification/ctrl.py           # 표현(raw IQ / dechirp) 별 SOM 대조군
verification/cfo_headroom.py   # 고정 CFO 손실 + headroom (§5.1)
verification/pkt_test.py       # 패킷 pooling vs genie (§5.2)
verification/why_gap.py        # repo detector vs MF 격차의 원인 분리 (§3.2)
verification/cheap_mf.py       # FFT-MF == direct MF 등가성 + 연산량 (§3.3)
verification/break_test.py     # 깊은 SNR 한계 + CFO drift (§5.3, §5.4)
```

## 폴더 구성

폴더는 **실험 시간 순**으로 번호가 매겨져 있으며, 이름에 그 시점의 접근이 드러난다.
`01 → 07` 순서로 읽으면 §3.1 까지의 실제 코드 궤적을, `verification/` 은 §3.2 이후 재검증의 궤적을 따라갈 수 있다.

```
LoRa-bam-reconstruction/
├── utils/                                 # 공용: LoRa 클래스, BAM 구현, 심볼 추정
├── 01_baseline_stft-magnitude-bam/        # STFT magnitude(dB) + MultiBAM
├── 02_bam-v3_complex-spectrogram/         # Complex 스펙트로그램 + MultiBAMv3
├── 03_bam-v4_tied-autoencoder/            # Tied AE 구조
├── 04_bam-v5_untied-residual-huber/       # Untied + outer residual + Huber
├── 05_bam-v6_residual-fc-blocks/          # Residual FC block 스택
├── 06_bam_complex-iq-direct/              # 단층 Complex-Valued BAM (IQ 직접 입력)
├── 07_noise2noise/                        # Noise2Noise 학습 (1차 결과)
├── verification/                          # 재검증 — 1차 결론 반증 + CFO headroom 측정
├── simulation_theory/                     # 복소수 활성함수(modReLU, zReLU) 시각화
└── local/                                 # README용 결과 이미지
```

공통 설정: SF = 9 (심볼 512개), fs = 1 MHz, 노이즈는 AWGN.

### 01_baseline_stft-magnitude-bam/ — 실수 도메인 baseline
- IQ → STFT → BW crop → **magnitude(dB) / magnitude flatten** 위에서 BAM 학습.
- 결과: 모든 SNR에서 baseline과 동률 또는 낮음. Complex 정보 소실이 원인으로 추정 → 다음 폴더에서 complex 도메인으로 이동.

### 02_bam-v3_complex-spectrogram/ — Complex 스펙트로그램 진입
- Complex STFT (real/imag concat 7936-dim) + `MultiBAMv3` (Torch, identity 활성, target-driven W 갱신, weight decay + grad clip).
- BW = 250 kHz, OSF = 4.
- **출력 collapse 관측** — 입력이 달라져도 출력이 상수 근처로 수렴, 정답률 0.0~0.4%.

### 03_bam-v4_tied-autoencoder/ — Tied AE로 재파라미터화
- `W_dec = W_enc^T` 로 묶고 leaky_relu + MSE + Adam.
- 같은 collapse 재현. **단일 심볼 디버깅에서 복원 peak이 clean 대비 0.2% 수준** → 진폭 붕괴가 학습 목적함수 수준의 문제임을 확인.

### 04_bam-v5_untied-residual-huber/ — Outer residual + Huber
- `x + α·δ` outer residual과 Huber loss로 collapse 방지.
- collapse는 해결. 그러나 모든 SNR에서 baseline보다 낮음 (예: -20 dB baseline 99.4% vs BAM 74.1%).

### 05_bam-v6_residual-fc-blocks/ — 깊이 확장
- Encoder/decoder를 `ResidualFCBlock` 스택으로 확장, `residual_alpha` epoch 스케줄.
- baseline과 거의 동률. 깊이만 늘려서는 개선되지 않음이 확인됨 → **여기서 "구조 실험" 라인 중단.**

### 06_bam_complex-iq-direct/ — 도메인 축소 (스펙트로그램 → IQ)
- 단층 **Complex-Valued BAM** (`ComplexIQBAM`). split-tanh + gain clamp + anchor.
- IQ를 직접 입력받아 dv/dt·du/dt Euler 동역학으로 복원.
- Complex-valued 가중치 자체의 표현력이 문제인지 검증하기 위한 축소 실험.
- **사후 평가:** §5.2 에 의해 이 구조는 CFO 여지에도 구조적으로 도달할 수 없다. 여지가 심볼 내부가 아니라 심볼들 사이에 있기 때문이며, 표현력이 아니라 관측 범위(심볼 1개)가 한계였다.

### 07_noise2noise/ — 최종 결과가 나온 지점
- 02와 동일 파이프라인(Complex 스펙트로그램 + `MultiBAMv3`)을 **Noise2Noise 자가지도**로 재학습.
- 아키텍처는 2-layer로 단축 (7936 → 2048 → 512), 학습 SNR U(-7.5, 0), 30 epochs.
- **§3.1의 결과표/그래프가 이 실험에서 나온 것.** 단, §3.2 에서 이 개선폭은 올바른 baseline 앞에서 유지되지 않음이 확인되었다.
- 당시에는 개선 상한을 이해하기 위해 SOM 분석으로 넘어갔고, 그 해석은 §4.1 에서 철회되었다.

### simulation_theory/
- modReLU, zReLU 복소수 활성함수의 magnitude/phase 시각화. 활성함수 선택의 근거로 사용.

### utils/
- `LoRa.py` — LoRa 심볼 생성/AWGN + BAM 계열 전 구현 (`BAM`, `MultiBAM`, `BAMv3`, `BAMv3_Huber`, `BAMv4`, `MultiBAMv5`, `MultiBAMv6`, `ComplexIQBAM`).
- `my_lora_utils.py` — `estimate_symbol_custom(a, sf, fs, bw)`: dechirp + FFT + **OSF-fold power 합산** 후 argmax.

## 부록 — 핵심 함수

원본 디코딩 코드에는 `power[:2^sf]` 로 앞 bin만 잘라 보는 오류가 있어 **OSF>1일 때 FFT power의 OSF-fold 합산이 누락**되어 있었다. 아래 함수는 이를 수정한 것으로, 01~07 모든 실험의 baseline·평가 파이프라인이 이 함수를 공유한다.

> **다만 이 수정 자체는 최적 detector가 아니다.** §3.2 에서 측정했듯 이 파이프라인의 dechirp+FFT는 심볼 에너지를 단일 bin에 모으지 못하고 인접 bin과 두 alias 블록으로 분산시킨다. OSF-fold는 그 분산된 조각을 non-coherent하게 일부 회수할 뿐이고, folding 개수를 바꿔도 격차는 남는다. 최적 검출기는 512개 프로토타입과의 상관(noncoherent matched filter)이며, 격차는 SNR 환산 약 5 dB 다. **01~07 의 모든 정확도 수치는 이 ≈5 dB suboptimal baseline 기준으로 읽어야 한다.**

```python
def estimate_symbol_custom(a, sf, fs, bw):
    # 1. signal_len 을 2^sf 정수배로 잘라 맞춤
    # 2. osf = signal_len // 2^sf
    # 3. down-chirp 생성 후 dechirp
    # 4. FFT → power = |spectrum|^2
    # 5. osf > 1 이면 power.reshape(osf, 2^sf).sum(axis=0) 로 OSF-fold 합산
    # 6. argmax → symbol index
```
