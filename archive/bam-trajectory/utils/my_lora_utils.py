
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, resample_poly, correlate, spectrogram, stft
import lzma
import pywt
import lz4.frame
from numpy.linalg import svd

sample_rate = 1e6   # 1 MHz
bw = 125e3          # LoRa Bandwidth (250 kHz)
sf = 9       # Spreading Factor 

symbol_time = 2**sf / bw  # Symbol duration
N = 2**sf
CARRIER = 915e6
PPM36 = 36/1e6 * CARRIER 
#Another Parameter
SNR = 0
USING_AWGN = True
nfft = 256
CFO_RX = 0
# Time vector
t = np.arange(0, symbol_time, 1/sample_rate)
samplePerDfreq = sample_rate / bw
samplePerSymbol = samplePerDfreq * (2**sf)

# Generate Downchirp (Linear Frequency Modulation)
f0 = -bw/2 # Start frequency
f1 = bw/2 # End frequency

###### Discrete-time chirp (normalized version)
k = np.arange(N)
B_k = np.exp(1j * 2 * np.pi * (k**2 / (2 * N) - k / 2))
###### Discrete-time chirp (normalized version)

# Generate Upchirp (increasing frequency)
up_chirp_signal = np.exp(1j * 2 * np.pi * (f0 * t + (f1 / ( symbol_time)) * t**2))

# Generate Downchirp signal (Linear Frequency Modulation)
down_chirp_signal = np.conj(up_chirp_signal)


def colortext(text,ncolor = 92):
    tetx = str(text)
    return f'\033[{ncolor}m' + tetx + "\033[0m"
# downsampled__ = resample_poly(dechirped, up=1, down=8)

def correction_factor_func(cfo):
    return np.exp(-1j * 2 * np.pi * cfo* t)

def calculate_power(signal):
    """Calculate Power given the signal"""
    total_power = np.mean(np.abs(signal) ** 2)
    #total_power = np.sum(np.abs(signal) ** 2)
    return total_power

# Q1: Implement cosine similarity function (10 pts)
def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    dot = np.dot(A,B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot / (norma * normb)
    ### END CODE HERE ###
    return cos

def generate_lora_chirp(symbol, t):
    """ Generate a LoRa upchirp for a given symbol index, ensuring frequency wraps around. """
    num_symbols = 2**sf # Total possible symbols
    symbol_freq_shift = symbol * samplePerDfreq  # Frequency shift for symbol
    
    final = []
    for index, row in enumerate(t):
        real_index = int(np.round(index + symbol_freq_shift))
       
        real_index = int(real_index % int(np.floor(( num_symbols* samplePerDfreq))))      
        final.append(np.exp(1j * 2 * np.pi * ((f0) * t[real_index] + (f1 / (symbol_time)) * t[real_index]**2)))
    # Generate chirp signal for the symbol
    return np.array(final, dtype=np.complex128)
    #return np.exp(1j * 2 * np.pi * ((f0 + symbol_freq_shift) * t + (bw / (2 * symbol_time)) * t**2))

def estimate_symbol(a, title, tresshold=0):
    """
    Estimate LoRa symbol from IQ samples using dechirp + FFT.
    Uses global variables: down_chirp_signal, sf
    
    Parameters:
        a: np.ndarray — complex I/Q samples (must match down_chirp_signal length)
        title: str — debug title (unused, kept for compatibility)
        tresshold: float — threshold for peak detection (unused)
    
    Returns:
        symbol: int — estimated symbol index (0 to 2^sf - 1)
        peak_power: float — peak power value at the detected symbol
    """
    if len(a) != len(down_chirp_signal):
        raise ValueError(f"Length mismatch: IQ={len(a)}, downchirp={len(down_chirp_signal)}")

    # Dechirp: multiply with conjugate downchirp
    dechirped = a * down_chirp_signal

    # FFT (no shift for standard LoRa processing)
    spectrum = np.fft.fft(dechirped)
    power = np.abs(spectrum)
    
    # Extract first 2^sf bins (LoRa symbol range)
    bins = 2 ** sf
    valid_power = power[:bins]
    
    # Find peak
    symbol = np.argmax(valid_power)
    peak_power = valid_power[symbol]
    
    return symbol, peak_power

import numpy as np

def estimate_symbol_custom(a, sf, fs, bw):
    """Estimate LoRa symbol index via dechirp + FFT.

    Notes:
    - This function assumes `a` contains (approximately) one LoRa symbol.
    - If `fs/bw` > 1 (OSF), the FFT bins are folded by OSF **in power**.
    """
    num_symbols = 2 ** sf
    signal_len = len(a)

    if signal_len < num_symbols:
        raise ValueError(f"Signal too short: len(a)={signal_len} < 2**sf={num_symbols}")

    # If length isn't an integer multiple of 2**sf, truncate to the nearest multiple.
    if signal_len % num_symbols != 0:
        new_len = (signal_len // num_symbols) * num_symbols
        if new_len <= 0:
            raise ValueError(f"Invalid signal length: len(a)={signal_len}")
        a = a[:new_len]
        signal_len = new_len

    osf = signal_len // num_symbols

    # Dechirp
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

    
def awgn(signal, snr_db):
    """Additive White Gaussian Noise (AWGN) to a signal."""
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))

    if USING_AWGN:
        return signal + noise
    else:
        return signal

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Applies a low-pass Butterworth filter to the data."""
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def PLOT_SPECGRAM(signal,nfft_,title,noverlap=32,rate=sample_rate):
    # Spectrogram for chirp detection
    plt.specgram(signal, NFFT=nfft_, noverlap=noverlap, Fs=rate, cmap="jet")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(label="Power")
    plt.ylim(-bw,bw)
    plt.yticks([-bw, -bw/2, -bw/4, 0, bw/4, bw/2,bw])  # Custom tick at 65250 Hz
    plt.show()

def PLOT_SPECGRAM2(data, NFFT, label, ax=None,noverlap=32,rate=sample_rate):
    if ax is None:
        ax = plt.gca()  # get current axis if not provided
    Pxx, freqs, bins, im = ax.specgram(data, NFFT=NFFT, Fs=rate, noverlap=noverlap, cmap='jet')
    # ax.specgram(data, NFFT=NFFT, Fs=rate, noverlap=noverlap, cmap='jet')
    ax.set_title(f"{label}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")

    return im

# # Center frequency axis (since return_onesided=False)
def show_spectrogram_from_npy(x):
    plt.figure(figsize=(8, 6))
    plt.imshow(x, aspect='auto', origin='lower', cmap='jet')
    plt.title("Spectrogram (sf=9, bw=125 kHz)")
    plt.xlabel("Time bins")
    plt.ylabel("Frequency bins")
    plt.colorbar(label="Normalized magnitude")
    plt.show()

def show_multiple_spectrograms(spec_list, titles=None, ncols=3):
    n = len(spec_list)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)  # Flatten even if 1D

    for i, ax in enumerate(axes):
        if i < n:
            im = ax.imshow(spec_list[i], aspect='auto', origin='lower', cmap='jet')
            # im = ax.imshow(spec_list[i], origin='lower', cmap='jet')
            ax.set_title(titles[i] if titles else f"Spectrogram {i+1}")
            ax.set_xlabel("Time bins")
            ax.set_ylabel("Frequency bins")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)   
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def findDominantFrequency(a):
    # Extract the first upchirp from received signal
   
    # Compute FFT of first received symbol
    fft_result = np.fft.fft(a)
    freqs = np.fft.fftfreq(len(a), 1 / sample_rate)
    
    # Find dominant frequency shift (coarse CFO estimate)
    peak_index = np.argmax(np.abs(fft_result))
    detected_freq = freqs[peak_index]
    # Expected frequency (assuming ideal upchirp peak is at BW/2)
    expected_freq = bw / 2  
    CFO_ = detected_freq 
    return CFO_

def correction_cfo_sto(rx_samples):
    
    oneSymbol = int(samplePerSymbol)
    total_buffer = 0
    i = 0
    dechirped_max = []
    fup = []
    fdown = []
    while total_buffer < len(rx_samples):
        frameBuffer = rx_samples[total_buffer:(total_buffer + oneSymbol)]
        total_buffer = total_buffer + oneSymbol
        
        ### DECHIRPED WITH DOWN CHIRP
        dechirp_a = frameBuffer * down_chirp_signal
        psd = np.fft.fftshift(np.fft.fft(dechirp_a))
        maxindex = np.argmax(psd)
        fup.append(maxindex)
        ### DECHIRPED WITH DOWN CHIRP

        ### DECHIRPED WITH UP CHIRP    
        dechirp_b = frameBuffer * up_chirp_signal
        psd = np.fft.fftshift(np.fft.fft(dechirp_b))
        maxindex = np.argmax(psd)   
        fdown.append(maxindex)
        ### DECHIRPED WITH UP CHIRP  

        ### FIND the maximum amplitude of dechirping with up chirp  
        maxAmplitude = np.max(np.abs(np.fft.fft(dechirp_b)))
        dechirped_max.append(maxAmplitude)

        i = i + 1

    # Compute mean , 
    # Find first index where value > mean
    mean_val = np.mean(dechirped_max)
    indices = np.where(dechirped_max > mean_val)[0]
    Index_that_start_a_down_chirp = indices[0] if len(indices) > 0 else None

    #### Then we will find 2 up chirp and 2 down chirp
    fup_chosen = Index_that_start_a_down_chirp - 2
    fdown_chosen = Index_that_start_a_down_chirp + 1

    CFO = (fup[fup_chosen] + fdown[fdown_chosen] )/(2*(N)) * bw
    # correction_factor = np.exp(-1j * 2 * np.pi * CFO* t)
    CFO = CFO % (bw/2)
    print(CFO)
    ### TEST TYPE 2
    test1 = rx_samples[(fup_chosen)*oneSymbol:(fup_chosen)*oneSymbol + oneSymbol] *  np.conj(np.exp(-1j * 2 * np.pi * CFO* t)) 
    test2 = rx_samples[(fup_chosen)*oneSymbol:(fup_chosen)*oneSymbol + oneSymbol] *  np.exp(-1j * 2 * np.pi * CFO* t) 
    corr = correlate(test1, up_chirp_signal, mode="full", method="fft")
    corr2 = correlate(test2, up_chirp_signal, mode="full", method="fft")
    
    corr_magnitude = np.abs(corr)**2    
    corr_magnitude2 = np.abs(corr2)**2
    
    max_corr = np.max(corr_magnitude)
    mean_corr = np.mean(corr_magnitude)
    max_corr2 = np.max(corr_magnitude2)
    mean_corr2 = np.mean(corr_magnitude2)
    
    dechirped_1 = rx_samples[(fup_chosen)*oneSymbol:(fup_chosen)*oneSymbol + oneSymbol] * up_chirp_signal
    dechirped_2 = rx_samples[(fup_chosen+1)*oneSymbol:(fup_chosen+1)*oneSymbol + oneSymbol] * up_chirp_signal
    phase_diff = np.angle(np.vdot(dechirped_2, dechirped_1))
    print(phase_diff)
   
    if (mean_corr2  >= mean_corr): # test 1 lebih berkolerasi , pilih test1
        CFO_FINAL = CFO   
        print("Turunin sinyal")  
        correction_factor = np.exp(-1j * 2 * np.pi * CFO_FINAL* t)
    else :
        CFO_FINAL = bw/2 - CFO
        print("Naikin sinyal")
        correction_factor = np.conj(np.exp(-1j * 2 * np.pi * CFO_FINAL* t))
    
    print("OUR CFO ESTIMATION IS : ",CFO_FINAL)
    if (np.abs(CFO_FINAL - bw/2) < 1000): #its weird
        CFO_FINAL = bw/2 - CFO_FINAL
        print("OUR CFO get edited IS : ",CFO_FINAL)
        if (max_corr > max_corr2):
            print("Naikin sinyal ke 2")
            correction_factor = np.conj(np.exp(-1j * 2 * np.pi * CFO_FINAL* t))
        else:
            print("Turunin sinyal ke 2")
            correction_factor = np.exp(-1j * 2 * np.pi * CFO_FINAL* t)
    print(CFO_RX)

    if (CFO_RX != 0):

        if ((np.abs(CFO_FINAL - np.abs(CFO_RX)) / np.abs(CFO_RX)) < (3/100)):
            print('SUCCESS , because error is',(np.abs(CFO_FINAL - np.abs(CFO_RX)) / np.abs(CFO_RX)) * 100, " %")
        else :
            print('FAIL , because error is',(np.abs(CFO_FINAL - np.abs(CFO_RX)) / np.abs(CFO_RX)) * 100, " %")
    rx_samples_corrected_cfo =  rx_samples[(fup_chosen-1)*oneSymbol:(fup_chosen-1)*oneSymbol + oneSymbol]* correction_factor

    corr = correlate(rx_samples_corrected_cfo, up_chirp_signal, mode="full", method="fft")
    peak_index = np.argmax(np.abs(corr))
    # STO_Correction = peak_index % oneSymbol
    peak_index = peak_index % oneSymbol
    print("Please adjust the window :",peak_index)
    a = len(rx_samples)
    b = len(correction_factor)
    correction_factor_10 = np.tile(correction_factor, int(a/b))
    rx_samples_corrected_sto_corrected_cfo = (rx_samples * correction_factor_10)[peak_index:]
    return rx_samples_corrected_sto_corrected_cfo,CFO_FINAL,peak_index

def Compress_IQ2(data):

    # Scale real and imag to int8 range [-128, 127]
    scaled_i = np.clip(np.real(data) * 127, -128, 127).astype(np.int8)
    scaled_q = np.clip(np.imag(data) * 127, -128, 127).astype(np.int8)

    # Interleave I and Q as [I0, Q0, I1, Q1, ...]
    iq_interleaved = np.empty(data.size * 2, dtype=np.int8)
    iq_interleaved[0::2] = scaled_i
    iq_interleaved[1::2] = scaled_q
    # Save to .raw
    with open("compressed_iq.raw", "wb") as f:
        iq_interleaved.tofile(f)

    return iq_interleaved

def UnCompress_IQ2(data):
    
    i = data[0::2].astype(np.float32) / 127  # De-quantize
    q = data[1::2].astype(np.float32) / 127
    data_reconstructed = i + 1j * q  # complex64
    return data_reconstructed

def Compress_lzma(data):
    # Compress
    raw_bytes = data.tobytes()
    compressed_bytes = lzma.compress(raw_bytes)
    return compressed_bytes

def UnCompress_lzma(data):
    decompressed_bytes = lzma.decompress(data)
    iq_decompressed = np.frombuffer(decompressed_bytes, dtype=np.complex64)
    return iq_decompressed

def Compress_lz4(data):
    # Compress
    raw_bytes = data.tobytes()
    compressed_bytes = lz4.frame.compress(raw_bytes)
    return compressed_bytes

def UnCompress_lz4(data):
    decompressed_bytes = lz4.frame.decompress(data)
    iq_decompressed = np.frombuffer(decompressed_bytes, dtype=np.complex64)
    return iq_decompressed

def Compress_IQ7bit(data):
    """
    Compress complex IQ data into 7-bit resolution using value range [-64, 63].
    Interleave I/Q and save to .raw as int8.
    """

    # Scale real and imag parts to fit into [-64, 63]
    scaled_i = np.clip(np.real(data) * 64, -64, 63).astype(np.int8)
    scaled_q = np.clip(np.imag(data) * 64, -64, 63).astype(np.int8)

    # Interleave I and Q as [I0, Q0, I1, Q1, ...]
    iq_interleaved = np.empty(data.size * 2, dtype=np.int8)
    iq_interleaved[0::2] = scaled_i
    iq_interleaved[1::2] = scaled_q

    # Save to binary file
    with open("compressed_iq.raw", "wb") as f:
        iq_interleaved.tofile(f)

    return iq_interleaved

def UnCompress_IQ7bit(data):
    """
    Decompress 7-bit compressed IQ data (int8) from range [-64, 63].
    Converts back to float32 complex IQ signal.
    """
    i = data[0::2].astype(np.float32) / 64.0  # De-quantize using the same scale
    q = data[1::2].astype(np.float32) / 64.0
    data_reconstructed = i + 1j * q
    return data_reconstructed

def Compress_IQ6bit(data):
    """
    Compress complex IQ data into 7-bit resolution using value range [-64, 63].
    Interleave I/Q and save to .raw as int8.
    """

    # Scale real and imag parts to fit into [-64, 63]
    scaled_i = np.clip(np.real(data) * 32, -32, 31).astype(np.int8)
    scaled_q = np.clip(np.imag(data) * 32, -32, 31).astype(np.int8)

    # Interleave I and Q as [I0, Q0, I1, Q1, ...]
    iq_interleaved = np.empty(data.size * 2, dtype=np.int8)
    iq_interleaved[0::2] = scaled_i
    iq_interleaved[1::2] = scaled_q

    # Save to binary file
    with open("compressed_iq.raw", "wb") as f:
        iq_interleaved.tofile(f)

    return iq_interleaved

def UnCompress_IQ6bit(data):
    """
    Decompress 7-bit compressed IQ data (int8) from range [-64, 63].
    Converts back to float32 complex IQ signal.
    """
    i = data[0::2].astype(np.float32) / 32.0  # De-quantize using the same scale
    q = data[1::2].astype(np.float32) / 32.0
    data_reconstructed = i + 1j * q
    return data_reconstructed
'''
Data 1 is decompression, data 2 is original
'''
def check_simmilarity(data1,data2): #data 2 suppose to original or desired

    # Load data
    # Ensure they are the same length
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]

    # Compute magnitude of errors
    errors = np.abs(data2 - data1)
    magnitudes = np.abs(data2)

    # Compute MSPE
    mspe = np.mean((errors) ** 2)
    rmse = np.sqrt(np.mean((np.abs(data1 - data2))**2))
    print('RMSE : ',rmse)
    print("RMPE : ", mspe)
    rmse_to_max_ratio_p = (rmse / (np.max(np.abs(data1)) + 1e-9)) * 100
    rmse_to_mean_ratio_p = (rmse / (np.mean(np.abs(data1)) + 1e-9)) * 100  # or mean, or std
    # Then use a rough rule of thumb:
    rmse_a=  colortext(format(rmse_to_max_ratio_p, ".3f") + " %")
    rmse_b=  colortext(format(rmse_to_mean_ratio_p, ".3f") + " %")
    print(f'RMSE - to signal max ratio = {rmse_a}')
    print(f'RMSE - to signal mean ratio = {rmse_b}')
    # NRMSE	Interpretation
    # < 0.05	Excellent match
    # 0.05–0.1	Good
    # 0.1–0.2	Acceptable
    # > 0.2	Poor similarity
     # Ensure same length

    # Compute EVM
    error = data1 - data2
    evm_rms = np.sqrt(np.sum(np.abs(error) ** 2) / np.sum(np.abs(data2) ** 2))
    evm_rms_percent = evm_rms * 100
    evm_rms_color=  colortext(format(evm_rms, ".3f"))
    evm_rms_percent_color=  colortext(format(evm_rms_percent, ".3f") + " %")
    print(f'Error Vector Magnitude = {evm_rms_color} => {evm_rms_percent_color}')
    
    return rmse

def Compress_IQ_8bit_scale(data):
    scaled_i = np.real(data)
    
    scaled_q = np.imag(data)
    
    iq_fusion = np.empty(data.size * 2, dtype=np.float32)
    iq_fusion [0::2] = scaled_i
    iq_fusion [1::2] = scaled_q
   
    x_min = iq_fusion.min()
    x_max = iq_fusion.max()
    x_scaled = (iq_fusion - x_min) / (x_max - x_min)      # → [0,1]
    
    x_normalized = x_scaled * 255 - 128           # → [-128,127]
    x_int8 = np.round(x_normalized).astype(np.int8)
    return x_int8,x_min,x_max

def deCompress_IQ_8bit_scale(data, x_min, x_max):
    # # Convert int8 to float in range [-128, 127] → [0,1] → rescale
    # i = ((data[0::2].astype(np.float32) + 128) / 255)     # [0,1]
    # q = ((data[1::2].astype(np.float32) + 128) / 255)

    # i_recon = i * (x_max - x_min) + x_min
    # q_recon = q * (x_max - x_min) + x_min

    # data_reconstructed = i_recon + 1j * q_recon
    x_scaled_back = (data.astype(np.float32) + 128) / 255

    # Recover original float range
    x_reconstructed = x_scaled_back * (x_max - x_min) + x_min

    # Split into I/Q
    i_recon = x_reconstructed[0::2]
    q_recon = x_reconstructed[1::2]
    data_reconstructed = i_recon + 1j * q_recon
    return data_reconstructed.astype(np.complex64)
    
def Compress_IQ_7bit_scale(data):
    scaled_i = np.real(data)
    scaled_q = np.imag(data)
    
    iq_fusion = np.empty(data.size * 2, dtype=np.float32)
    iq_fusion [0::2] = scaled_i
    iq_fusion [1::2] = scaled_q
   
    x_min = iq_fusion.min()
    x_max = iq_fusion.max()
    x_scaled = (iq_fusion - x_min) / (x_max - x_min)      # → [0,1]
    x_normalized = x_scaled * 127 - 64           # → [-128,127]
    x_int8 = np.round(x_normalized).astype(np.int8)
    return x_int8,x_min,x_max

def deCompress_IQ_7bit_scale(data, x_min, x_max):
    x_scaled_back = (data.astype(np.float32) + 64) / 127

    # Recover original float range
    x_reconstructed = x_scaled_back * (x_max - x_min) + x_min

    # Split into I/Q
    i_recon = x_reconstructed[0::2]
    q_recon = x_reconstructed[1::2]
    data_reconstructed = i_recon + 1j * q_recon
    return data_reconstructed.astype(np.complex64)

def Compress_IQ_6bit_scale(data):
    scaled_i = np.real(data)
    scaled_q = np.imag(data)
    
    iq_fusion = np.empty(data.size * 2, dtype=np.float32)
    iq_fusion [0::2] = scaled_i
    iq_fusion [1::2] = scaled_q
   
    x_min = iq_fusion.min()
    x_max = iq_fusion.max()
    x_scaled = (iq_fusion - x_min) / (x_max - x_min)      # → [0,1]
    x_normalized = x_scaled * 63 - 32           # → [-128,127]
    x_int8 = np.round(x_normalized).astype(np.int8)
    with open("compressed_iq_sean.raw", "wb") as f:
        x_int8.tofile(f)
    return x_int8,x_min,x_max

def deCompress_IQ_6bit_scale(data, x_min, x_max):
    x_scaled_back = (data.astype(np.float32) + 32) / 63

    # Recover original float range
    x_reconstructed = x_scaled_back * (x_max - x_min) + x_min

    # Split into I/Q
    i_recon = x_reconstructed[0::2]
    q_recon = x_reconstructed[1::2]
    data_reconstructed = i_recon + 1j * q_recon
    return data_reconstructed.astype(np.complex64)

def Compress_IQ_5bit_scale(data):
    scaled_i = np.real(data)
    scaled_q = np.imag(data)
    
    iq_fusion = np.empty(data.size * 2, dtype=np.float32)
    iq_fusion [0::2] = scaled_i
    iq_fusion [1::2] = scaled_q
   
    x_min = iq_fusion.min()
    x_max = iq_fusion.max()
    x_scaled = (iq_fusion - x_min) / (x_max - x_min)      # → [0,1]
    x_normalized = x_scaled * 31 - 16           # → [-128,127]
    x_int8 = np.round(x_normalized).astype(np.int8)
    with open("compressed_iq_sean.raw", "wb") as f:
        x_int8.tofile(f)
    return x_int8,x_min,x_max

def deCompress_IQ_5bit_scale(data, x_min, x_max):
    x_scaled_back = (data.astype(np.float32) + 16) / 31

    # Recover original float range
    x_reconstructed = x_scaled_back * (x_max - x_min) + x_min

    # Split into I/Q
    i_recon = x_reconstructed[0::2]
    q_recon = x_reconstructed[1::2]
    data_reconstructed = i_recon + 1j * q_recon
    return data_reconstructed.astype(np.complex64)
     
def Compress_IQ_4bit_scale(data):
    scaled_i = np.real(data)
    scaled_q = np.imag(data)
    
    iq_fusion = np.empty(data.size * 2, dtype=np.float32)
    iq_fusion [0::2] = scaled_i
    iq_fusion [1::2] = scaled_q
   
    x_min = iq_fusion.min()
    x_max = iq_fusion.max()
    x_scaled = (iq_fusion - x_min) / (x_max - x_min)      # → [0,1]
    x_normalized = x_scaled * 15 - 8           # → [-128,127]
    x_int8 = np.round(x_normalized).astype(np.int8)
    with open("compressed_iq_sean2.raw", "wb") as f:
        x_int8.tofile(f)
    return x_int8,x_min,x_max

def deCompress_IQ_4bit_scale(data, x_min, x_max):
    x_scaled_back = (data.astype(np.float32) + 8) / 15

    # Recover original float range
    x_reconstructed = x_scaled_back * (x_max - x_min) + x_min

    # Split into I/Q
    i_recon = x_reconstructed[0::2]
    q_recon = x_reconstructed[1::2]
    data_reconstructed = i_recon + 1j * q_recon
    return data_reconstructed.astype(np.complex64)
        
def Compress_IQ_3bit_scale(data):
    scaled_i = np.real(data)
    scaled_q = np.imag(data)
    
    iq_fusion = np.empty(data.size * 2, dtype=np.float32)
    iq_fusion [0::2] = scaled_i
    iq_fusion [1::2] = scaled_q
   
    x_min = iq_fusion.min()
    x_max = iq_fusion.max()
    x_scaled = (iq_fusion - x_min) / (x_max - x_min)      # → [0,1]
    x_normalized = x_scaled * 7 - 4           # → [-128,127]
    x_int8 = np.round(x_normalized).astype(np.int8)
    with open("compressed_iq_sean2.raw", "wb") as f:
        x_int8.tofile(f)
    return x_int8,x_min,x_max

def deCompress_IQ_3bit_scale(data, x_min, x_max):
    x_scaled_back = (data.astype(np.float32) + 4) / 7

    # Recover original float range
    x_reconstructed = x_scaled_back * (x_max - x_min) + x_min

    # Split into I/Q
    i_recon = x_reconstructed[0::2]
    q_recon = x_reconstructed[1::2]
    data_reconstructed = i_recon + 1j * q_recon
    return data_reconstructed.astype(np.complex64)

def Compress_IQ_2bit_scale(data):
    scaled_i = np.real(data)
    scaled_q = np.imag(data)
    
    iq_fusion = np.empty(data.size * 2, dtype=np.float32)
    iq_fusion [0::2] = scaled_i
    iq_fusion [1::2] = scaled_q
   
    x_min = iq_fusion.min()
    x_max = iq_fusion.max()
    x_scaled = (iq_fusion - x_min) / (x_max - x_min)      # → [0,1]
    x_normalized = x_scaled * 3 - 2           # → [-128,127]
    x_int8 = np.round(x_normalized).astype(np.int8)
    with open("compressed_iq_sean2.raw", "wb") as f:
        x_int8.tofile(f)
    return x_int8,x_min,x_max

def deCompress_IQ_2bit_scale(data, x_min, x_max):
    x_scaled_back = (data.astype(np.float32) + 2) / 3
    # Recover original float range
    x_reconstructed = x_scaled_back * (x_max - x_min) + x_min
    # Split into I/Q
    i_recon = x_reconstructed[0::2]
    q_recon = x_reconstructed[1::2]
    data_reconstructed = i_recon + 1j * q_recon
    return data_reconstructed.astype(np.complex64)

def complexThresh(testSesh, coefs, multiplier, upsampling_factor=8):
    #thrR = ddencmp(numpy.real(testSesh)) * multiplier
    thrR = multiplier
    for i in range(len(coefs)):
        coefs[i] = pywt.threshold(coefs[i], thrR, 'hard')
    return

def DWT_compress(ActiveSession, multiplier):
    coefs = pywt.wavedec(ActiveSession, 'db5', level=5)
    complexThresh(ActiveSession, coefs, multiplier)

    levels = [len(x) for x in coefs]
    levels = np.asarray(levels, dtype=np.float32)

    compressThis = np.ndarray(0, dtype=np.complex64)
    for i in range(len(coefs)):
        compressThis = np.append(compressThis, coefs[i])
    compressThis = compressThis.view(np.float32)
    lz4_compressed = Compress_lz4(compressThis)
    return (lz4_compressed, levels)

def DWT_decompress(lz4_compressed, level_info, wavelet='db5', dtype=np.float32):
    # Step 1: LZ4 decompress
    decompressed_bytes = lz4.frame.decompress(lz4_compressed)
    
    # Step 2: Convert bytes back to NumPy array
    float_array = np.frombuffer(decompressed_bytes, dtype=dtype)
    complex_array = float_array.view(np.complex64)

    # Step 3: Split back into wavelet coefficient arrays
    levels = level_info.astype(int)
    coefs = []
    idx = 0
    for length in levels:
        coefs.append(complex_array[idx:idx + length])
        idx += length
    # Step 4: Inverse wavelet transform to get IQ signal
    reconstructed = pywt.waverec(coefs, wavelet)

    # Step 5: Ensure output is complex64
    return reconstructed.astype(np.complex64)

def svd_compresssion(data,m,t=None):
    n_complex = len(data) // m  # integer division
    M_complex = data[:m*n_complex].reshape((m, n_complex))
    M_real = np.empty((m, 2*n_complex), dtype=np.float32)
    M_real[:, 0::2] = M_complex.real.astype(np.float32)
    M_real[:, 1::2] = M_complex.imag.astype(np.float32)
    # --- compute SVD (full thin SVD) ---
    U, s, Vt = svd(M_real, full_matrices=False)  # s sorted descending
    energy = (s**2).cumsum() / (s**2).sum()
    L_=0
    if t is None:
        L_ = np.searchsorted(energy, 0.99) + 1   # keep >=99% energy
    else:
        L_ = t
    # or set t explicitly: t = 20
    Ut = U[:, :L_].astype(np.float16)
    st = s[:L_].astype(np.float16)
    Vtt = Vt[:L_, :].astype(np.float16)
    return Ut,st,Vtt

def svd_decompresssion(Ut,st,Vtt,m,n_complex):
    M_approx = (Ut * st) @ Vtt  # broadcasting st across columns of Ut
    M_rec_complex = M_approx.reshape(m, 2*n_complex)
    # metrics
    
    I = M_rec_complex[:, 0::2]
    Q = M_rec_complex[:, 1::2]
    # Combine into complex64
    reconstructed_complex64_flat = (I + 1j*Q).astype(np.complex64).ravel()
    return reconstructed_complex64_flat

def downsampling(data,fs,down):
    #Example for Downsampling 
    # Example input signal of length 4096
    down = down
    fs_new = fs/down
    y = resample_poly(data, up=1, down=down)
    return y,fs_new
    # Resample from 4096 → 1024 using polyphase filtering

def create_spectrogram_npy(x_ds,fs_ds,snr,symbol,no,folder=None):
    nperseg = 128
    noverlap = 64
    nfft = 512
    f, t, Sxx = spectrogram(
        x_ds, fs=fs_ds, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, mode='psd', return_onesided=False
    )
    
    ###### USE DB, and normalize
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
   
    Sxx_dB = 10 * np.log10(Sxx + 1e-12)
    # Normalize to 0–1 range
    Sxx_norm = (Sxx_dB - Sxx_dB.min()) / (Sxx_dB.max() - Sxx_dB.min())
    # ---- CROP to ±BW/2 ----
    mask = (f >= -bw/2) & (f < bw/2)
    Sxx_crop = Sxx_norm[mask, :]
    if (folder is not None):
        np.save(f'{folder}/s_sf9_bw125_{snr}_{symbol}_{no}.npy', Sxx_crop)
    return Sxx_crop,Sxx_dB.min(),Sxx_dB.max()

def create_spectrogram_npy_dual(x_ds,fs_ds,snr,symbol,no,folder_r=None,folder_i=None):
    nperseg = 128 #128
    noverlap = 64 # 64
    nfft = 512 #512
    window = "hann"
    f_new, t_new, Zxx = stft(
        x_ds, fs=fs_ds, window=window, nperseg=nperseg,
        noverlap=noverlap, nfft=nfft, padded=True, boundary="zeros",
        return_onesided=False
    )

    
    ###### USE DB, and normalize
    f = np.fft.fftshift(f_new)
    Zxx_shift = np.fft.fftshift(Zxx, axes=0)
    # Sxx_dB = 10 * np.log10(Sxx + 1e-12)
    # Normalize to 0–1 range
    # Sxx_norm_0_to_1 = (Zxx_shift - Zxx_shift.min()) / (Zxx_shift.max() - Zxx_shift.min())
    Zxx_r = np.real(Zxx_shift)
    # Zxx_r = np.abs(Zxx_r)
    Zxx_i = np.imag(Zxx_shift)
    # Zxx_i = np.abs(Zxx_i)
    Zxx_r_norm_0_to_1 = (Zxx_r - Zxx_r.min()) / (Zxx_r.max() - Zxx_r.min())
    Zxx_i_norm_0_to_1 = (Zxx_i - Zxx_i.min()) / (Zxx_i.max() - Zxx_i.min())
    # Sxx_norm_0_to_1 = (Zxx_shift - Zxx_shift.min()) / (Zxx_shift.max() - Zxx_shift.min())
    # Zxx_r_norm = 2 * (Zxx_r - Zxx_r.min()) / (Zxx_r.max() - Zxx_r.min()) - 1
    # Zxx_i_norm = 2 * (Zxx_i - Zxx_i.min()) / (Zxx_i.max() - Zxx_i.min()) - 1
    # ---- CROP to ±BW/2 ----
    mask = (f >= -bw/2) & (f < bw/2)
    Zxx_r_crop = Zxx_r_norm_0_to_1[mask, :]
    Zxx_i_crop = Zxx_i_norm_0_to_1[mask, :]
    
    if (folder_r is not None) and (folder_i is not None):
        np.save(f'{folder_r}/s_sf9_bw125_{snr}_{symbol}_{no}.npy', Zxx_r_crop)
        np.save(f'{folder_i}/s_sf9_bw125_{snr}_{symbol}_{no}.npy', Zxx_i_crop)
    return Zxx_r_crop,Zxx_i_crop,Zxx_r.min(),Zxx_r.max(),Zxx_i.min(),Zxx_i.max()

def calculate_symbol_alliqfile_without_down_sampling(data,sf,bw,sample_rate,show=True):
    plt.figure(figsize=(20,25))
    symbol_time = 2**sf / bw  # Symbol duration
    osr = int(np.floor(sample_rate/bw))
    # Time vector
    t = np.arange(0, symbol_time, 1/sample_rate)
    # Generate Downchirp (Linear Frequency Modulation)
    f0 = -bw/2 # Start frequency
    f1 = bw/2 # End frequency
    ###### Discrete-time chirp (normalized version)
    result = []
    array_of_value = []
    # Generate Upchirp (increasing frequency)
    up_chirp_signal = np.exp(1j * 2 * np.pi * (f0 * t + (f1 / (symbol_time)) * t**2))
    # Generate Downchirp signal (Linear Frequency Modulation)
    down_chirp_signal = np.conj(up_chirp_signal)

    nsamp = int(2**sf/bw*sample_rate) # 4096 ? 512 ?
    n = len(data)
    frames = n // nsamp
    num_chunks = int(np.ceil(len(data) / nsamp))
    padded_data = np.pad(data, (0, (num_chunks * nsamp) - len(data) + (1 * nsamp)), mode='constant')
    for i in range(frames):
        start_idx = i
        end_idx = i+1
        bias = 0#2824 - 1432 # -1400 # -540
        an_data = padded_data[start_idx*nsamp + bias :end_idx*nsamp + bias]
        de_ = an_data * down_chirp_signal
        ## EXPERIMENTAL 1
        # de_fft = np.fft.fftshift(np.fft.fft(de_))
        # de_fft_abs = np.abs(de_fft)**2
        # de_fft_argmax = np.argmax(de_fft_abs)
        ## EXPERIMENTAL 2
        # spec = np.fft.fft(de_, n=(2**sf)*osr)
        # power = np.abs(spec)**2
        # power_512 = power.reshape(2**sf, osr).sum(axis=1)
        # symbol = int(np.argmax(power_512))
        # EXPERIMENTAL 3
        spectrum = np.fft.fftshift(np.fft.fft(de_))
        power = np.abs(spectrum) ** 2
        fft_len = len(power)
        center = fft_len // 2
        bins = 2 ** sf 
        upper_freq = power[center : center + bins]
        lower_freq = power[center - bins: center]  
        combine = upper_freq + lower_freq
        symbol = np.argmax(combine)
        array_of_value.append(np.max(combine))
        ## END EXPERIMENTAL
        result.append(symbol)
        if (show):
            plt.subplot(7,6,i+1)
            # plt.specgram(an_data)
            plt.plot(combine)
            plt.title(f'Peak : {symbol}')
    if (show):
        plt.show()

    ## Post Processing
    # Example array

    # Calculate average
    avg = sum(array_of_value) / len(array_of_value)
    # Set threshold: 10% below average
    threshold = avg * 0.4

    # Check each value
    tags = ["LOW" if x < threshold else "OK" for x in array_of_value]

    return result, tags

def calculate_symbol_alliqfile_with_down_sampling(data,sf,bw,sample_rate,show=True):

    plt.figure(figsize=(20,25))

    data_downsampling = resample_poly(data, up=bw, down=sample_rate)

    symbol_time = 2**sf / bw  # Symbol duration
    # Time vector
    t = np.arange(0, symbol_time, 1/bw)
    
    f0 = -bw/2 # Start frequency
    f1 = bw/2 # End frequency
    result = []
    # Generate Upchirp (increasing frequency)
    up_chirp_signal = np.exp(1j * 2 * np.pi * (f0 * t + (f1 / (symbol_time)) * t**2))
    # Generate Downchirp signal (Linear Frequency Modulation)
    down_chirp_signal = np.conj(up_chirp_signal)

    nsamp = int(2**sf) # 4096 ? 512 ?
    n = len(data_downsampling)
    frames = n // nsamp
    num_chunks = int(np.ceil(len(data_downsampling) / nsamp))
    padded_data = np.pad(data_downsampling, (0, (num_chunks * nsamp) - len(data_downsampling)), mode='constant')
    for i in range(frames):
        start_idx = i
        end_idx = i+1
        an_data = padded_data[start_idx*nsamp:end_idx*nsamp]
        de_ = an_data * down_chirp_signal
        de_fft = np.fft.fft(de_)
        de_fft_abs = np.abs(de_fft)
        de_fft_argmax = np.argmax(de_fft_abs)
        result.append(de_fft_argmax)
        if (show):
            plt.subplot(7,6,i+1)
            plt.plot(de_fft_abs)
            plt.title(f'Peak : {de_fft_argmax}')
    if (show):
        plt.show()
    return result

def quantize_iq(x, nbits=8):
    # Normalize to [-1,1] first (choose appropriate scale)
    xmax = np.max(np.abs(x))
    if xmax == 0: return x
    scale = (2**(nbits-1)-1) / xmax
    xq = np.round(x * scale) / scale
    return xq

# example: 500 Hz CFO
# x_cfo = apply_cfo(x, Fs=1e6, freq_offset_hz=500)
def apply_cfo(x, Fs, freq_offset_hz):
    n = np.arange(len(x))
    return x * np.exp(1j * 2 * np.pi * freq_offset_hz * n / Fs)

# example: linewidth 100 Hz
# x_pn = apply_phase_noise(x, Fs=1e6, linewidth_hz=100)
def apply_phase_noise(x, Fs, linewidth_hz):
    # approximate phase noise as Wiener process with PSD ~ linewidth    
    n = len(x)
    dt = 1.0 / Fs
    sigma = np.sqrt(2 * np.pi * linewidth_hz * dt)  # per-sample increment std
    increments = np.random.normal(0, sigma, n)
    phase = np.cumsum(increments)  # random walk
    return x * np.exp(1j * phase)

from scipy.signal import fftconvolve

# delays = [0, int(1e-6*1e6), int(3e-6*1e6)]  # e.g., 0,1,3 microsec at Fs=1MHz
# x_mp = multipath_rayleigh(signal, delays)
def multipath_rayleigh(x, delays_samples, K_dB = 6):
    # taps: complex gains array, delays_samples: integer delays
    #### create taps
    num_taps = 3
    K = 10**(K_dB/10)
    # LOS component amplitude sqrt(K/(K+1))
    los_amp = np.sqrt(K/(K+1))
    scatter_std = np.sqrt(1/(2*(K+1)))
    taps = los_amp + (np.random.randn(num_taps) + 1j*np.random.randn(num_taps))*scatter_std

    ####
    # build impulse response
    L = max(delays_samples) + 1
    h = np.zeros(L, dtype=complex)
    for g, d in zip(taps, delays_samples):
        h[d] = g
    return fftconvolve(x, h)[:len(x)]

# # apply e.g., fd 10 Hz
# x_tv = time_varying_rayleigh(x, fd_hz=10, Fs=1e6)
def time_varying_rayleigh(x, fd_hz, Fs):
    n = len(x)
    dt = 1/Fs
    # approximate correlation time 1/fd; create complex Gaussian noise then lowpass
    white = (np.random.randn(n) + 1j*np.random.randn(n)) / np.sqrt(2)
    # simple exponential smoothing for correlation
    alpha = np.exp(-2*np.pi*fd_hz*dt)
    h = np.zeros(n, dtype=complex)
    h[0] = white[0]
    for i in range(1,n):
        h[i] = alpha*h[i-1] + np.sqrt(1-alpha**2)*white[i]
    # normalize
    h = h / np.sqrt(np.mean(np.abs(h)**2))
    return x * h

def apply_iq_imbalance(x, gain_imb_db=0.5, phase_imb_deg=2.0):
    g = 10**(gain_imb_db/20.0)
    phi = np.deg2rad(phase_imb_deg)
    i = np.real(x)
    q = np.imag(x)
    i2 = g*i*np.cos(0) - g*q*np.sin(phi)
    q2 = i*np.sin(phi) + q*np.cos(0)
    return i2 + 1j*q2

# hard clip amplitude > clip_level
def hard_clip(x, max_amp=0.8):
    mags = np.abs(x)
    phases = np.angle(x)
    mags_clipped = np.minimum(mags, max_amp)
    return mags_clipped * np.exp(1j*phases)

from scipy.signal import butter, lfilter
# # e.g., 100-200 kHz interferer band
# x_col = band_limited_noise(x, Fs=1e6, low_hz=100e3, high_hz=200e3, snr_db=10)
def band_limited_noise(x, Fs, low_hz, high_hz, snr_db):
    n = len(x)
    white = (np.random.randn(n) + 1j*np.random.randn(n)) / np.sqrt(2)
    b, a = butter(4, [low_hz/(Fs/2), high_hz/(Fs/2)], btype='band')
    colored = lfilter(b, a, white)
    # scale to desired SNR relative to signal
    Psig = np.mean(np.abs(x)**2)
    Pn = np.mean(np.abs(colored)**2)
    scale = np.sqrt(Psig / (10**(snr_db/10) * Pn))
    return x + colored * scale