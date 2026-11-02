import numpy as np

def estimate_symbol_custom(a, sf, fs, bw):
    """Estimate LoRa symbol index via dechirp + FFT.

    For OSF>1 (fs/bw), fold FFT bins by summing power over OSF blocks.
    """
    num_symbols = 2 ** sf
    signal_len = len(a)

    if signal_len < num_symbols:
        raise ValueError(f"Signal too short: len(a)={signal_len} < 2**sf={num_symbols}")

    if signal_len % num_symbols != 0:
        new_len = (signal_len // num_symbols) * num_symbols
        if new_len <= 0:
            raise ValueError(f"Invalid signal length: len(a)={signal_len}")
        a = a[:new_len]
        signal_len = new_len

    osf = signal_len // num_symbols

    t = np.arange(signal_len) / fs
    f0 = -bw / 2
    symbol_time = num_symbols / bw
    phase = 2 * np.pi * (f0 * t + (bw / (2 * symbol_time)) * t**2)
    down_chirp = np.conj(np.exp(1j * phase))

    dechirped = a * down_chirp
    spectrum = np.fft.fft(dechirped)
    power = np.abs(spectrum) ** 2

    if osf > 1:
        folded_power = power.reshape(osf, num_symbols).sum(axis=0)
        symbol = int(np.argmax(folded_power))
        peak_power = float(folded_power[symbol])
    else:
        valid_power = power[:num_symbols]
        symbol = int(np.argmax(valid_power))
        peak_power = float(valid_power[symbol])

    return symbol, peak_power

def run_simulation_demo():
    # 설정
    sf = 9
    bw = 125000
    osf = 8
    fs = bw * osf
    num_symbols = 2**sf
    snr_db = -20  # 테스트 목표 SNR
    trials = 1000
    
    print(f"Simulation Start: SF{sf}, SNR {snr_db}dB, Trials {trials}")

    correct_count = 0
    
    # Noise scaling
    # IMPORTANT: With oversampling (fs=bw*osf), "SNR in BW" differs from "SNR per sample"
    # by +10*log10(osf) because noise is spread across fs while signal occupies ~bw.
    # If you want SNR_BW = snr_db, then the per-sample noise variance must be scaled by osf.
    snr_bw_db = snr_db
    snr_sample_db = snr_bw_db + 10 * np.log10(osf)

    sig_power = 1.0
    noise_power = sig_power / (10**(snr_sample_db / 10))
    noise_std = np.sqrt(noise_power / 2)

    for i in range(trials):
        # 1. 랜덤 심볼 생성 및 변조 (Vectorized)
        target = np.random.randint(0, num_symbols)
        
        # Tx Signal 생성
        symbol_time = num_symbols / bw
        t = np.arange(int(symbol_time * fs)) / fs
        
        # 주파수: -BW/2에서 시작하여 BW만큼 증가 (target만큼 shift)
        # 핵심: (Base + Shift) % BW - BW/2 형태로 랩핑 구현
        slope = bw / symbol_time
        f_inst_raw = (slope * t + target * (bw / num_symbols)) % bw
        f_inst = f_inst_raw - bw/2
        
        # 위상 적분 (Phase Accumulation)
        dt = 1/fs
        phase = 2 * np.pi * np.cumsum(f_inst) * dt
        tx_sig = np.exp(1j * phase)
        
        # 2. 노이즈 추가
        noise = noise_std * (np.random.randn(len(tx_sig)) + 1j * np.random.randn(len(tx_sig)))
        rx_sig = tx_sig + noise
        
        # 3. 복조 (수정된 Folding 함수 사용)
        est_symbol, _ = estimate_symbol_custom(rx_sig, sf, fs, bw)
        
        if est_symbol == target:
            correct_count += 1
            
    acc = correct_count / trials * 100
    print(f"Result Accuracy: {acc:.2f}%")

# 위에 정의한 estimate_symbol_final 함수가 있어야 합니다.
run_simulation_demo()