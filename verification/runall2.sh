set -e
for s in 7 8 9 10; do
  echo "=== SF$s as-run (그들 add_noise 규약 통일, U=10) ==="
  PYTHONIOENCODING=utf-8 python -u nelora_asrun.py ../NeLoRa_Dataset/$s --sf $s --n 1800 --U 10 --boot 150
done
echo "=== 완료 ==="
