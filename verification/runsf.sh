set -e
for s in 9 10; do
  echo "=== SF$s as-run (U=10, SF 보정 격자) ==="
  PYTHONIOENCODING=utf-8 python -u nelora_asrun.py ../NeLoRa_Dataset/$s --sf $s --n 1800 --U 10 --boot 0
done
echo "=== 완료 ==="
