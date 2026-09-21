set -e
echo "=== (2a) SF7 U=10 검증: U=100 판(-15.31)과 맞는지 + 잡음재추출 부트스트랩 ==="
PYTHONIOENCODING=utf-8 python -u nelora_asrun.py ../NeLoRa_Dataset/7 --sf 7 --n 3000 --U 10 --boot 200
for s in 8 9 10; do
  echo
  echo "=== (3) SF$s as-run (U=10, 점추정) ==="
  PYTHONIOENCODING=utf-8 python -u nelora_asrun.py ../NeLoRa_Dataset/$s --sf $s --n 1800 --U 10 --boot 0
done
echo "=== 완료 ==="
