set -e
for s in 7 8 9 10; do
  echo "=== SF$s as-run (그들 규약 + 배치 단위 잡음, U=10, batch=16) ==="
  PYTHONIOENCODING=utf-8 python -u nelora_asrun.py ../NeLoRa_Dataset/$s --sf $s --n 1800 --U 10 --batch 16 --boot 150
done
echo "=== 완료 ==="
