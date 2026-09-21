# -*- coding: utf-8 -*-
"""
그들 main.py 에 '대역 필터 입력' 옵션을 붙이는 최소 패치.

왜 필요한가 — 리뷰어의 가장 강한 반론:
  "데시메이션 한 단계가 신경망보다 2.1 dB 더 번다" 는 둘을 대체재처럼 놓는다.
  그러나 둘은 함께 쓸 수 있다. 공정한 비교는 **DNN 에도 같은 대역 필터를 준 것**이다.
  데시메이션 + DNN 이 표준 수신기를 넘는가?

본편 README §4.2 (AWGN 에서 심볼 단위 front-end 는 정합필터를 넘을 수 없다) 가
답을 예측한다: 대역 밖 잡음을 먼저 지운 입력으로 학습하면 DNN 의 이득은
표준 수신기 대비 거의 0 이어야 한다.

무엇을 바꾸는가:
  add_noise() 가 잡음을 더한 뒤, ±BW/2 브릭월 LPF 를 통과시킨다.
  **OSF=8 과 텐서 모양(num_samples)은 그대로 둔다** — 데시메이션까지 하면 입력 크기가
  바뀌어 아키텍처를 고쳐야 하므로, 대역 밖 잡음만 지우고 표본화율은 유지한다.
  이러면 drop-in 이고 에포크당 비용도 같다.

  LPF 는 dataX(잡음 있는 쪽)에만 건다. dataY(깨끗한 목표)는 그대로 두면 마스크가
  '대역 밖을 지우는 법' 까지 배워야 하므로, 목표에도 같은 필터를 건다.

사용법 (WSL, 그들 저장소 안에서):
  python nelora_lpf_patch.py main.py            # main_lpf.py 생성
  python main_lpf.py --sf 7 --batch_size 64     # ckpt_sf7_lpf/ 에 저장

  # 평가 (내 하네스도 같은 LPF 를 걸어야 한다)
  python nelora_dnn_eval.py --sf 7 --batch 16 --lpf \
      --data-dir ~/LoRa/data/7/ --cache ~/LoRa/cache_sf7.npz \
      --ckpt-dir ckpt_sf7_lpf --boot 60 --out result_sf7_lpf.json
"""
import io, sys, os

LPF_FN = '''

# ---- 대역 필터 (추가) : ±BW/2 브릭월. OSF 와 텐서 모양은 그대로 둔다.
def brickwall_lpf(x):
    """x: (..., num_samples) complex tensor. 대역 밖 잡음만 지운다."""
    X = torch.fft.fft(x, dim=-1)
    keep = num_classes // 2
    mask = torch.zeros(num_samples, dtype=torch.bool, device=x.device)
    mask[:keep] = True
    mask[-(num_classes - keep):] = True
    return torch.fft.ifft(X * mask, dim=-1)
'''


def main(src, dst=None):
    dst = dst or os.path.join(os.path.dirname(src) or '.',
                              os.path.basename(src).replace('.py', '_lpf.py'))
    s = io.open(src, encoding='utf-8').read()

    # 1) LPF 함수 삽입 (num_classes / num_samples 정의 뒤)
    anchor = "# define models"
    assert anchor in s, 'main.py 구조가 예상과 다르다'
    s = s.replace(anchor, LPF_FN.strip('\n') + '\n\n\n' + anchor, 1)

    # 2) 체크포인트 저장 위치 분리 (원본 덮어쓰기 방지)
    s = s.replace("save_ckpt_dir = f'ckpt_sf{sf}'",
                  "save_ckpt_dir = f'ckpt_sf{sf}_lpf'")

    # 3) add_noise 끝에서 dataX, dataY 에 같은 LPF 를 건다
    old = """    if normalization:
        dataX = dataX / torch.mean(torch.abs(dataX)) # normalization
    return dataX, dataY, truth_idx"""
    new = """    dataX = brickwall_lpf(dataX)          # <-- 추가: 대역 밖 잡음 제거
    dataY = brickwall_lpf(dataY)          # <-- 목표도 같은 대역으로
    if normalization:
        dataX = dataX / torch.mean(torch.abs(dataX)) # normalization
    return dataX, dataY, truth_idx"""
    assert old in s, 'add_noise 구조가 예상과 다르다'
    s = s.replace(old, new, 1)

    # 4) 학습만 돌도록
    i = s.rindex("if __name__ == '__main__':")
    s = s[:i] + "if __name__ == '__main__':\n    train()\n"

    io.open(dst, 'w', encoding='utf-8').write(s)
    print(f'생성: {dst}')
    print('  - brickwall_lpf() 추가 (±BW/2, OSF=8 유지)')
    print('  - add_noise() 가 dataX/dataY 에 LPF 적용')
    print(f'  - 체크포인트는 ckpt_sf<SF>_lpf/ 로 분리')
    print('  - train() 만 실행')
    print('\n주의: 캐시(pkl_<SF>.pkl)는 LPF 와 무관하므로 지울 필요 없다.')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'main.py')
