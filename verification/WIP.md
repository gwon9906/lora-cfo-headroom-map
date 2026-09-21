# 진행 중 — 다음 세션이 여기부터 읽으면 된다

최종 갱신 2026-09-22 · 전체 결과는 `../README_NELORA.md`, 검토 패킷은 `../REVIEW_NELORA_2026-09-21.md`

---

## 지금 어디까지 왔나

**확정된 것** — SF7 / BW125k / 10% SER / 그들 조건 / 같은 배치 (`nelora_dnn_eval.py`,
그들 `test()` 무수정 실행과 네 지점 0.1 dB 이내 일치):

```
decode_loraphy (U=100)        -13.11 dB   기준 0
NELoRa DNN (100에포크 재학습)   -14.49 dB   +1.39 dB
표준 RX (학습·정렬 없음)        -16.57 dB   +3.46 dB
                                           ---------
표준 RX - DNN                 +2.08 dB    (macro 로는 +2.24 dB)
```

**held-out(패킷 분리 18패킷)으로도 쟀다.** DNN 이 +1.3 dB 로 그들 발표 SF7(+1.46)에
근접하므로, "as-run 이라 DNN 에 유리했다" 는 단서는 더 이상 필요 없다.
표준 RX -16.2, D2 정렬 -16.9, **D3 정렬 -17.1**.

---

## 지금 돌고 있는 것

WSL 에서 `main_fair.py --epochs 400 --lpf` (약 2시간 15분). 150에포크 판은 수렴하지
않았다(검증 곡선이 +0.031 %p/에포크로 상승 중, 최고가 144/150). 끝나면:

1. 곡선 확인 — 마지막 50에포크 기울기가 ±0.01 안쪽이면 수렴
2. **두 팔 모두 검증 최고 에포크**로 재평가 (지금 표는 둘 다 최종 에포크라
   과적합한 plain 팔에 불리하다). `--ckpt-mask ckpt_sf7_X/NNN_maskCNN.pkl` 로 지정
3. SF8/9/10 as-run 재측정 (`runall3.sh`) — 잡음 규약 수정 후 미실시. 학습과 동시에
   돌리면 메모리 부족으로 죽는다

## 이미 닫힌 것 — 대역제한 입력 검출기 (§5-10)

DNN 과 무관하게 완결됐다. 모두에게 ±BW/2 대역제한 입력을 주면
`decode_loraphy` 만 −12.98 → −16.02 (−3.03) 로 움직이고, 표준 RX(+0.00)·D2/D3(+0.03)
는 그대로다. 격차가 **+3.33 [+2.67, +4.02] → +0.28 [−0.14, +0.70] (0 과 구별 안 됨)**.
§5-1 의 분해를 독립적인 두 번째 방법으로 확인한 것.

## 남은 실험 하나 — §5-9 대역필터 입력 DNN

리뷰어의 첫 반론("데시메이션과 신경망은 대체재가 아니라 함께 쓸 수 있다")에 답하는 자리.
**판정 기준은 결과를 보기 전에 `README_NELORA.md` §5-9 에 박아 뒀다. 그대로 따를 것.**

| DNN(필터 입력) 도달 위치 | 해석 |
|---|---|
| 표준 RX(-16.2) 근처 | 정리와 일치. 정렬조차 못 배움 |
| 표준 RX ~ D3(-17.1) 사이 | 정리와 일치. **정렬 학습분** — 디노이징 이득 아님 |
| D3 초과 (CI 하한이 -17.1 위) | 정리와 충돌. 본편 §4.2 전제 점검 |

### WSL 에서 돌릴 것

```bash
cd ~/LoRa/NeLoRa_Dataset          # 그들 저장소 클론본
# 이 저장소에서 두 파일만 복사하면 된다 (외부 의존 없음)
cp .../verification/nelora_dnn_eval.py .
cp .../verification/nelora_lpf_patch.py .

python nelora_lpf_patch.py main.py            # main_fair.py 생성
# 두 팔을 같은 시드·같은 패킷 분할·같은 에포크로, 순차로 (동시 실행 금지)
nohup sh -c 'python main_fair.py --sf 7 --batch_size 64 --epochs 150 > train_plain.log 2>&1 && \
             python main_fair.py --sf 7 --batch_size 64 --epochs 150 --lpf > train_lpf.log 2>&1' &

# 평가 (held-out 만, D2/D3 기준선 포함)
python nelora_dnn_eval.py --sf 7 --batch 16 --max-symbols 0 --boot 100 \
  --data-dir ~/LoRa/data/7/ --cache ~/LoRa/cache_sf7_full.npz \
  --ckpt-dir ckpt_sf7_plain --only-idx split_sf7_test.npy --out result_plain.json
python nelora_dnn_eval.py --sf 7 --batch 16 --max-symbols 0 --boot 100 --lpf \
  --data-dir ~/LoRa/data/7/ --cache ~/LoRa/cache_sf7_full.npz \
  --ckpt-dir ckpt_sf7_lpf --only-idx split_sf7_test.npy --out result_lpf.json
```

결과 JSON 은 `verification/results/` 에 넣고 `nelora_dnn_report.py` 로 읽는다.

**귀무 결과(≈0)가 나와도 그것만으로는 증명이 아니다.** 학습곡선
(`curve_sf7_{plain,lpf}.csv`)이 평탄해졌음을 함께 보여야 한다. 아직 오르는 중이면
`--epochs` 를 늘린다. 여유가 되면 `--scratch` 판도 대조한다.

---

## 그 외 남은 것

- **SF8/9/10 as-run 재측정** — 잡음 규약 두 버그(아래)를 고친 뒤 SF7 만 다시 쟀다.
  `verification/runall3.sh` 를 돌리면 되는데, WSL 학습과 동시에 돌리면 메모리 부족으로
  죽는다. 학습이 끝난 뒤에 할 것.
- 브릭월을 실제 FIR(`scipy.signal.firwin`, 64~128탭)로 바꿔 "+3 dB 는 상한에 가깝다" 를
  측정값으로 만들기 (선택).
- 학습 프로세스가 한 번 겹쳐 돈 적이 있다(`TEST ACC` 104줄). 지금 `main_fair.py` 는
  체크포인트 디렉터리가 분리돼 있어 재발하지 않는다.

---

## 이 프로젝트에서 반복된 실패 양상 — 반드시 읽을 것

**틀린 것은 예외 없이 "내가 재구현한 쪽"이었다.** 지금까지 일곱 건:

1. `ideal_chirps` 가 위상을 `cumsum` 으로 적분 → OSF 별로 다른 파형 (`nelora_chirp` 로 교체)
2. `d1_decimate` 가 de-chirp 를 먼저 하고 대역을 잘랐다 (신호를 자름)
3. 그들 `add_noise` 의 정규화를 "argmax 에 무관" 이라며 생략 → 신경망 입력이 분포 밖
4. `.eval()` 을 호출 → 그들 `test()` 는 호출하지 않는다 (BatchNorm 이 배치 통계를 씀)
5. `amp` 를 심볼별로 계산 (그들은 배치 전체) → 곡선이 2.2 dB 낙관적
6. 잡음을 평가셋 전체에 한 번만 (그들은 DataLoader 배치마다) → CI 폭발
7. 심볼을 패킷 순으로 읽음 (그들은 코드 순) → **held-out 인덱스가 다른 심볼을 가리킴**

7번은 `IndexError` 가 우연히 드러냈다. 아니었으면 에러 없이 틀린 답이 나왔다.

### 그래서 지키는 규칙

- 새 검출기 팔은 **이상 신호 온전성 검사**(타이밍/CFO=0 에서 정합필터와 붙는가)를 먼저 통과시킨다.
- 남의 수치와 비교하기 전에 **그 수치를 낸 코드를 읽는다.**
- **재구현이 맞는지는 재구현끼리 비교해 알 수 없다. 원본을 돌려서 대조한다.**
  (그들 `test()` 무수정 실행이 3·4번을 잡았다.)
- 패치 생성물·인라인 구현은 **넘기기 전에 원본과 수치 대조한다.**
  (인라인 6함수를 원본 모듈과 대조해 오차 0 확인한 것이 그 예.)

---

## 그들 소스 (scratchpad 는 세션마다 사라지므로 다시 받을 것)

```bash
git clone --depth 1 https://github.com/daibiaoxuwu/NeLoRa_Dataset.git    # Bench, 우리가 쓰는 릴리스
git clone --depth 1 https://github.com/AIoT-MLSys-Lab/NELoRa.git         # SenSys, MATLAB 평가 행렬
```

SenSys 쪽 `neural_enhanced_demodulation/matlab/evaluation/*.mat` 에 그들 SF7 곡선이 있다
(`error_matrix` 는 이름과 달리 **정확도**다 — `evaluation.m` 이 `1-error_matrix` 를 그린다).
체크포인트·데이터셋은 Bench README 의 Google Drive 링크.
