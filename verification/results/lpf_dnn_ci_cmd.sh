#!/bin/bash
# §5-b 기준에 고정된 조건. 두 모드를 같은 조건으로 연속 실행한다.
# 그들 model_components 는 PYTHONPATH 로 붙인다 — 실행 디렉터리를 그들 저장소로
# 옮기면 provenance 의 git 필드가 그들 저장소를 가리켜 오해를 만든다.
set -u
cd ~/LoRa/run7
P=$HOME/miniconda3/envs/ai/bin/python
CK=$HOME/LoRa/NeLoRa_Dataset/ckpt_sf7_lpf
export PYTHONPATH=$HOME/LoRa/NeLoRa_Dataset
for MODE in eval train; do
  echo "=== model-mode=$MODE 시작 $(date +%H:%M:%S) ==="
  "$P" nelora_dnn_eval.py \
    --sf 7 --batch 16 --max-symbols 0 --noise-reals 10 --boot 1000 \
    --lpf --model-mode "$MODE" \
    --data-dir "$HOME/LoRa/data/7/" --cache "$HOME/LoRa/cache_sf7_full.npz" \
    --ckpt-mask "$CK/398_maskCNN.pkl" --ckpt-cls "$CK/398_C_XtoY.pkl" \
    --only-idx "$HOME/LoRa/NeLoRa_Dataset/split_sf7_test.npy" \
    --snrs=-10,-12,-14,-15,-16,-17,-18,-19,-20,-22,-24 \
    --out "lpf_dnn_ci_$MODE.json" > "lpf_dnn_ci_$MODE.txt" 2>&1
  echo "  exit=$? $(date +%H:%M:%S) -> lpf_dnn_ci_$MODE.{json,txt}"
done
echo ALLDONE
