import numpy as np
import numpy.matlib
from scipy.signal import chirp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from scipy.io import savemat
from scipy import signal

import torch

class LoRa:
    def __init__(self, sf, bw, OSF=8):
        """
        sf : spreading factor
        bw : bandwidth (Hz)
        OSF: oversampling factor (fs = bw * OSF)
        """
        self.sf = sf
        self.bw = bw
        self.OSF = OSF
        self.fs = int(bw * OSF)   # ★ 전체 시스템에서 쓸 샘플링 주파수

    def gen_symbol(self, code_word, down=False, Fs=None):
        sf = self.sf
        bw = self.bw

        # ★ 기본 샘플링 주파수는 self.fs = bw * OSF
        if Fs is None or Fs <= 0:
            Fs = self.fs

        # 이론적 심볼 시간 Ts
        Ts = (2 ** sf) / bw

        # 심볼 하나에 해당하는 샘플 수
        Ns = int(round(Ts * Fs))

        # 시간 축
        t = np.arange(Ns) / Fs

        # 기준 upchirp (baseband, -BW/2 ~ +BW/2 sweep)
        f0 = -bw / 2.0
        f1 = +bw / 2.0
        k = (f1 - f0) / Ts

        phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
        up_chirp = np.exp(1j * phase)

        # 데이터 심볼 m을 주파수 쉬프트로 실어줌
        m = int(code_word) % (2 ** sf)
        # 톤 주파수: f_m = m/Ts
        f_m = m / Ts
        phase_m = 2 * np.pi * f_m * t
        data_tone = np.exp(1j * phase_m)

        if not down:
            s = up_chirp * data_tone   # upchirp + data
        else:
            # downchirp = conj(upchirp) * data
            down_chirp = np.conj(up_chirp)
            s = down_chirp * data_tone

        return s.astype(np.complex128)


    def gen_symbol_exp(self, code_word, down=False):
        sf = self.sf
        bw = self.bw

        f_offset = bw/(2**sf) * code_word
        t_fold = (2**sf - code_word) / bw
        T = 2**sf/bw
        t1 = np.arange(0, t_fold, 1/bw)
        t2 = np.arange(t_fold, (2**sf)/bw, 1/bw)

        x1 = np.exp(1j*2*np.pi*(bw/(2*T)*(t1**2) + (f_offset - bw/2)*t1))
        x2 = np.exp(1j*2*np.pi*(bw/(2*T)*(t2**2) + (f_offset - 3*bw/2)*t2))
        result = np.concatenate((x1,x2),axis=0)
        if down:
            result = np.conj(result)
        return result
    
    def get_fft(self, signal):
        sig_fft = np.fft.fft(signal)
        return sig_fft
    
    def get_fft_abs(self, signal):
        sig_fft = self.get_fft(signal)
        sig_fft_abs = np.abs(sig_fft)
        return sig_fft_abs


    def plot_spectrogram(self, signal, noverlap, nfft):
        if noverlap is None and nfft is None:
            noverlap = 2**self.sf // 8
            nfft = 2**self.sf // 4
        plt.figure(figsize=(8,8))
        plt.specgram(signal, NFFT=nfft, noverlap=noverlap,Fs=self.bw)
        plt.show()
    
    def one_rows_two_cols(self, signal1, signal2, noverlap, nfft):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
        # 서브플롯들 사이의 간격을 조정
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.text(0.5, 0.04, 'Frequency index', ha='center')
        plt.suptitle('Spectrogram of two symbols')
        fig.text(0.04, 0.5, 'Frequency', rotation='vertical')

        formatter = FormatStrFormatter('%.3f')  # 소수점 2자리로 제한하는 포맷 설정
        ax1.xaxis.set_major_formatter(formatter)
        ax2.xaxis.set_major_formatter(formatter)

        # plt.subplot(1,2,1)
        ax1.specgram(signal1, NFFT=nfft, noverlap=noverlap, Fs=self.bw)
        # plt.subplot(1,2,2)
        ax2.specgram(signal2, NFFT=nfft, noverlap=noverlap, Fs=self.bw)
        plt.show()
    
    def plot_fft_real(self, signal):
        x = np.arange(len(signal))
        sig_fft = self.get_fft(signal)
        plt.scatter(x, sig_fft.real, c='#1e88e5',alpha=0.7)
        plt.plot(x, sig_fft.real, c='red', linestyle='dashed', alpha=0.5)
        plt.show()

    def plot_fft_imag(self, signal):
        x = np.arange(len(signal))
        sig_fft = self.get_fft(signal)
        plt.scatter(x, sig_fft.imag, c='#1e88e5',alpha=0.7)
        plt.plot(x, sig_fft.imag, c='red', linestyle='dashed', alpha=0.5)
        plt.show()

    def plot_fft_total(self, signal):
        x = np.arange(len(signal))
        sig_fft = self.get_fft(signal)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
        # 서브플롯들 사이의 간격을 조정
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.text(0.5, 0.04, 'Frequency index', ha='center')
        fig.text(0.08, 0.45, 'Magnitude', rotation='vertical')
        # plt.subplot(1,2,1)
        ax1.set_title('Real Part')
        ax1.scatter(x, sig_fft.real, c='#1e88e5',alpha=0.7)
        ax1.plot(x, sig_fft.real, c='red', linestyle='dashed', alpha=0.5)

        # plt.subplot(1,2,2)
        ax2.set_title('Imaginary Part')
        ax2.scatter(x, sig_fft.imag, c='#1e88e5',alpha=0.7)
        ax2.plot(x, sig_fft.imag, c='red', linestyle='dashed', alpha=0.5)

        plt.show()

    def awgn(self, signal_, SNR_):
        sig_avg_pwr = np.mean(abs(signal_)**2)
        sig_avg_db = 10*np.log10(sig_avg_pwr)
        noise_avg_db = sig_avg_db - SNR_
        noise_avg_pwr = 10**(noise_avg_db/10)
        noise_sim = np.random.normal(0, np.sqrt(noise_avg_pwr), len(signal_))
        return signal_ + noise_sim
    
    def awgn_iq(self, signal_, SNR_):
        sig_avg_pwr = np.mean(abs(signal_)**2)      # 신호의 평균 파워
        noise_avg_pwr = sig_avg_pwr / (10**(SNR_/10))   # SNR을 고려한 노이즈 파워 계산

        # if np.isrealobj(signal_):
        #     # 평균 : 0, 표준편차 : np.sqrt(noise_avg_pwr), 데이터 수: len(signal_)
        #     noise_sim = np.random.normal(0, np.sqrt(noise_avg_pwr), len(signal_))

        # else:
        noise_sim = (np.random.normal(0, np.sqrt(noise_avg_pwr/2), len(signal_)) + 1j*np.random.normal(0, np.sqrt(noise_avg_pwr/2), len(signal_)))

        return signal_ + noise_sim
    
    # SNR에 따른 실제 가우시안 노이즈 추가 방식 및 SNR 계산
    def add_awgn_noise(self, signal, snr_db):
        """주어진 SNR(dB)에 맞게 AWGN 노이즈 추가"""
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        return signal + noise
    
    def calculate_snr_db(self, clean_signal, noisy_signal):
        signal_power = np.mean(np.abs(clean_signal)**2)
        noise_power = np.mean(np.abs(noisy_signal - clean_signal)**2)
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db
    
    def generate_symbol_with_noise(self, sf, bw, generate_size, root_path, target_snr):
        lora_init = LoRa(sf, bw)
        sym_count = 0
        sym_index = 0
        for i in range(generate_size):
            val = i % int(2**sf)
            chirp = lora_init.gen_symbol_fs(val, i+7, bw, down=False, Fs=int(8*bw))
            gen_snr = target_snr
            # chirp_awgn = lora_init.add_awgn_noise(chirp, gen_snr)
            chirp_awgn = lora_init.awgn_iq(chirp, gen_snr)
            chirp_signal = chirp_awgn.reshape(1,-1)
            mat_data = {
            '__header__': b'Generating LoRa Symbol using gen_symbol()',
            '__version__': '1.0',
            '__globals__': [],
            'chirp': chirp_signal
            }
            if sym_count == (int(2**sf)):
                sym_index += 1
                sym_count = 0
            save_name = f'{sym_index}_{gen_snr}_{sf}_{bw}_0_{val}_0_0.mat'
            savemat(root_path + save_name, mat_data)
            sym_count += 1

    def generate_symbol_with_noise2(self, sf, bw, generate_size, root_path, target_snr):
        lora_init = LoRa(sf, bw)
        sym_count = 0
        sym_index = 0
        for i in range(generate_size):
            val = i % int(2**sf)
            chirp_ = lora_init.gen_symbol(val,down=False)
            chirp = signal.resample_poly(chirp_,up=8,down=1)
            gen_snr = target_snr
            # chirp_awgn = lora_init.add_awgn_noise(chirp, gen_snr)
            chirp_awgn = lora_init.awgn_iq(chirp, gen_snr)
            chirp_signal = chirp_awgn.reshape(1,-1)
            mat_data = {
            '__header__': b'Generating LoRa Symbol using gen_symbol()',
            '__version__': '1.0',
            '__globals__': [],
            'chirp': chirp_signal
            }
            if sym_count == (int(2**sf)):
                sym_index += 1
                sym_count = 0
            save_name = f'{sym_index}_{gen_snr}_{sf}_{bw}_0_{val}_0_0.mat'
            savemat(root_path + save_name, mat_data)
            sym_count += 1
    
    def fft_example(self, val):
        signal = self.gen_symbol_exp(val, sf=self.sf, down=False, Fs=self.bw)
        self.plot_fft_total(signal)
    
    def fft_example(self, val):
        signal = self.gen_symbol_exp(val, sf=self.sf, down=False, Fs=self.bw)
        self.plot_fft_total(signal)

    def gen_symbol_fs(self, code_word, sf, bw, down=False, Fs=None):
        # 파라미터 사용 (self 값 사용 안 함)
        # sf = self.sf
        # bw = self.bw
        # Fs = bw
        # the default sampling frequency is 1e6
        if Fs is None or Fs < 0:
            Fs = 1000000
        # bandwidth : use parameter value
        org_Fs = Fs

        # For Nyquist Theory
        if Fs < bw:
            Fs = bw
        
        t = np.arange(0, 2**sf/bw, 1/Fs)
        # print('len t : ', len(t))
        num_samp = Fs * 2**sf/bw

        f0 = -bw/2
        f1 = bw/2

        # chirpI = chirp(t, f0, 2**sf/bw, f1, 'linear', 90)
        # chirpQ = chirp(t, f0, 2**sf/bw, f1, 'linear', 0)
        chirpI = chirp(t, f0, 2**sf/bw, f1, 'linear', 0)
        chirpQ = chirp(t, f0, 2**sf/bw, f1, 'linear', -90)
        baseline = chirpI + 1j * chirpQ

        if down:
            baseline = np.conj(baseline)
        baseline = numpy.matlib.repmat(baseline,1,2)
        offset = round((2**sf - code_word) / 2**sf * num_samp)

        symb = baseline[:, int(num_samp - offset):int(num_samp - offset+int(num_samp))]

        if org_Fs != Fs:
            overSamp = int(Fs / org_Fs)
            symb = symb[:, ::overSamp]

        return symb[0]       
"""
NOTE 
This is the first structure of BAM and Multi BAM
not support for batch training
"""
class BAM:
    def __init__(self, input_dim, output_dim, eta=1e-4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eta = eta

        # 가중치 초기화 (입력 -> 출력)
        self.W = np.random.uniform(-0.01, 0.01, (output_dim, input_dim))

    def _output_function(self, Wx):
        return Wx  # 선형 활성화 함수

    def train(self, X):
        for i, x in enumerate(X):
            x = x.reshape(1, -1)

            # Forward pass (입력 -> 출력 -> 입력)
            y = self._output_function(self.W @ x.T)
            x_reconstructed = self._output_function(self.W.T @ y)

            # 재구성 오류 계산
            error = x - x_reconstructed.T

            # Hebbian 학습 규칙 수정: 재구성 오류를 최소화하도록 가중치 업데이트
            self.W += self.eta * np.outer(y, error)
            # self.W += self.eta * (y @ error)
            # self.W += self.eta * (y @ error)  

            if np.isnan(self.W).any():
                raise ValueError("NaN detected in weights! Check learning rate or initialization.")

    def compress(self, X):
        compressed = []
        for x in X:
            y = self._output_function(self.W @ x.T)
            compressed.append(y.T)
        return np.array(compressed)

    def decompress(self, compressed_X):
        decompressed = []
        for y in compressed_X:
            y = y.reshape(-1, 1)
            x_reconstructed = self._output_function(self.W.T @ y)
            decompressed.append(x_reconstructed.T)
        return np.array(decompressed)
"""
NOTE 
This is the first structure of BAM and Multi BAM
not support for batch training
"""
class MultiBAM:
    def __init__(self, layers_dims, eta=1e-4):
        self.bams = [
            BAM(layers_dims[i], layers_dims[i + 1], eta)
            for i in range(len(layers_dims) - 1)
        ]

    def train(self, X):
        for i, bam in enumerate(self.bams):
            bam.train(X)
            X = bam.compress(X)

    def compress(self, X):
        for bam in self.bams:
            X = bam.compress(X)
        return X

    def decompress(self, X):
        for bam in reversed(self.bams):
            X = bam.decompress(X)
        return X         
"""
NOTE Bam V2
This is the same structure of BAM and Multi BAM as V1
but support for batch training
"""
class BAMv2:
    def __init__(self, input_dim, output_dim, eta=1e-5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eta = eta

        # 가중치 초기화 (입력 -> 출력)
        self.W = np.random.uniform(-0.01, 0.01, (output_dim, input_dim))

    def _output_function(self, Wx):
        return Wx  # 선형 활성화 함수

    def train(self, X, num_epochs=1, batch_size=32, verbose=True):
        n_samples = X.shape[0]
        losses = []

        for epoch in range(num_epochs):
            perm = np.random.permutation(n_samples)
            X = X[perm]

            for i in range(0, n_samples, batch_size):
                batch = X[i:i+batch_size]
                batch_errors = []

                for x in batch:
                    x = x.reshape(1, -1)
                    y = self._output_function(self.W @ x.T)
                    x_reconstructed = self._output_function(self.W.T @ y)

                    error = x - x_reconstructed.T
                    batch_errors.append(np.mean(error**2))

                    self.W += self.eta * np.outer(y, error)

                    if np.isnan(self.W).any():
                        raise ValueError("NaN detected in weights!")

                # average error for this batch
                batch_mse = np.mean(batch_errors)
                losses.append(batch_mse)

                if verbose and i % (batch_size * 10) == 0:
                    print(f"Epoch {epoch+1}, Batch {i//batch_size+1}, MSE = {batch_mse:.6f}")

        return losses

    def compress(self, X):
        compressed = []
        for x in X:
            y = self._output_function(self.W @ x.T)
            compressed.append(y.T)
        return np.array(compressed)

    def decompress(self, compressed_X):
        decompressed = []
        for y in compressed_X:
            y = y.reshape(-1, 1)
            x_reconstructed = self._output_function(self.W.T @ y)
            decompressed.append(x_reconstructed.T)
        return np.array(decompressed)
"""
NOTE Bam V2
This is the same structure of BAM and Multi BAM as V1
but support for batch training
"""    
class MultiBAMv2:
    def __init__(self, layers_dims, eta=1e-4):
        self.bams = [
            BAMv2(layers_dims[i], layers_dims[i + 1], eta)
            for i in range(len(layers_dims) - 1)
        ]

    def train(self, X, num_epochs=1, batch_size=32):
        all_losses = []

        for i, bam in enumerate(self.bams):
            print(f"\n--- Training Layer {i+1}/{len(self.bams)} ---")
            losses = bam.train(X, num_epochs=num_epochs, batch_size=batch_size)
            all_losses.append(losses)
            X = bam.compress(X)  # feed compressed output to next layer

        return all_losses

    def compress(self, X):
        for bam in self.bams:
            X = bam.compress(X)
        return X

    def decompress(self, X):
        for bam in reversed(self.bams):
            X = bam.decompress(X)
        return X         

## NOTE Add GPU Processing wih Torch
####################################
"""
NOTE Bam V3
This is the same structure of BAM and Multi BAM V1/V2
support for batch training
And the most importantly, add torch instead of classical numpy
its increase the speed while maintain the performace
"""
class BAMv3:
    def __init__(self, input_dim, output_dim, eta=1e-5, device=None, max_update_norm=1.0, weight_decay=1e-4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eta = eta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_update_norm = max_update_norm
        self.weight_decay = weight_decay

        self.W = torch.empty(output_dim, input_dim, device=self.device)
        torch.nn.init.uniform_(self.W, -0.01, 0.01)

    def _output_function(self, Wx):
        return Wx

    def train(self, X_in, X_target, num_epochs=1, batch_size=64, verbose=True):
        X_in = torch.tensor(X_in, dtype=torch.float32, device=self.device)
        X_tg = torch.tensor(X_target, dtype=torch.float32, device=self.device)

        n_samples = X_in.shape[0]
        losses = []

        for epoch in range(num_epochs):
            perm = torch.randperm(n_samples, device=self.device)
            X_in = X_in[perm]
            X_tg = X_tg[perm]

            for i in range(0, n_samples, batch_size):
                xb = X_in[i:i+batch_size]     # (B, in)
                tb = X_tg[i:i+batch_size]     # (B, in)

                Xb = xb.T                      # (in, B)
                Tb = tb.T                      # (in, B)

                Y  = self.W @ Xb               # (out, B)
                Xh = self.W.T @ Y              # (in, B)

                E  = Tb - Xh                   # ✅ target 기준 error

                # 배치 평균 업데이트
                update = self.eta * (Y @ E.T) / xb.shape[0]   # (out, in)

                # update norm clipping
                u_norm = torch.norm(update)
                if u_norm > self.max_update_norm:
                    update = update * (self.max_update_norm / (u_norm + 1e-12))

                # weight decay (폭발 방지)
                self.W = (1.0 - self.weight_decay) * self.W + update

                # loss 기록 (target 기준)
                batch_mse = torch.mean((Tb - Xh) ** 2).item()
                losses.append(batch_mse)

                if verbose:
                    print(f"Epoch {epoch+1}, Batch {i//batch_size+1}/{(n_samples+batch_size-1)//batch_size}, MSE={batch_mse:.6f}")

        return losses

    def compress(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = (self.W @ X.T).T
        return y.detach().cpu().numpy()

    def decompress(self, Y):
        Y = torch.tensor(Y, dtype=torch.float32, device=self.device)
        x = (self.W.T @ Y.T).T
        return x.detach().cpu().numpy()
    
class MultiBAMv3:
    def __init__(self, layers_dims, eta=1e-5, device=None, max_update_norm=1.0, weight_decay=1e-4):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bams = [
            BAMv3(layers_dims[i], layers_dims[i + 1],
                  eta=eta, device=self.device,
                  max_update_norm=max_update_norm,
                  weight_decay=weight_decay)
            for i in range(len(layers_dims) - 1)
        ]

    def train(self, X_noisy, X_clean, num_epochs=1, batch_size=64):
        all_losses = []

        for li, bam in enumerate(self.bams):
            print(f"\n--- Training Layer {li+1}/{len(self.bams)} ---")
            losses = bam.train(X_noisy, X_clean, num_epochs=num_epochs, batch_size=batch_size, verbose=True)
            all_losses.append(losses)

            # 다음 레이어 입력/타깃 모두 같은 방식으로 압축
            X_noisy = bam.compress(X_noisy)
            X_clean = bam.compress(X_clean)

        return all_losses

    def compress(self, X):
        for bam in self.bams:
            X = bam.compress(X)
        return X

    def decompress(self, X):
        for bam in reversed(self.bams):
            X = bam.decompress(X)
        return X


import torch
import numpy as np
import matplotlib.pyplot as plt

class BAMv3_Huber:
    """
    Single Layer BAM with Huber Loss & Vectorized Batch Processing
    """
    def __init__(self, input_dim, output_dim, eta=1e-4, huber_delta=0.01, device=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eta = eta
        self.huber_delta = huber_delta
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 가중치 초기화
        self.W = torch.empty(output_dim, input_dim, device=self.device)
        torch.nn.init.uniform_(self.W, -0.01, 0.01)

    def _output_function(self, Wx):
        # LoRa 실험에서는 identity 그대로 두고, 필요하면 tanh로 바꿔도 됨
        return Wx  

    def train(self, X, num_epochs=1, batch_size=32, verbose=False):
        # 입력을 Tensor로 정규화
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device, dtype=torch.float32)
        
        n_samples = X.shape[0]
        losses = []

        for epoch in range(num_epochs):
            # Shuffle (원본 X는 유지, 에폭별 view에만 적용)
            perm = torch.randperm(n_samples, device=self.device)
            X_epoch = X[perm]

            for i in range(0, n_samples, batch_size):
                batch = X_epoch[i:i+batch_size]        # (B, In)
                if batch.shape[0] == 0:
                    continue
                B = batch.shape[0]

                # 1. Forward (Vectorized)
                Y = self._output_function(batch @ self.W.T)   # (B, Out)
                
                # 2. Reconstruction
                X_reconstructed = self._output_function(Y @ self.W)  # (B, In)
                
                # 3. Error
                raw_error = batch - X_reconstructed          # (B, In)
                
                # 4. Huber Gradient Logic
                abs_error = torch.abs(raw_error)
                mse_region = abs_error <= self.huber_delta

                # grad용 effective error (Huber의 dL/dx 부분)
                effective_error = torch.where(
                    mse_region,
                    raw_error,
                    self.huber_delta * torch.sign(raw_error)
                )

                # 5. Loss Calculation (모니터링용, per-element 기준)
                loss_mse = 0.5 * (raw_error[mse_region] ** 2)
                loss_mae = self.huber_delta * (abs_error[~mse_region] - 0.5 * self.huber_delta)
                # 전체 element 개수로 나눠서 scale 맞추기
                num_elems = raw_error.numel()
                batch_loss = (loss_mse.sum() + loss_mae.sum()) / num_elems
                losses.append(batch_loss.item())

                # 6. Weight Update (batch 평균으로 정규화)
                with torch.no_grad():
                    grad = (Y.T @ effective_error) / B       # (Out, In)
                    self.W += self.eta * grad

                if torch.isnan(self.W).any():
                    raise ValueError("NaN detected in weights!")

            if verbose:
                print(f"[Epoch {epoch+1}/{num_epochs}] last batch Huber loss: {batch_loss.item():.6f}")

        return losses

    def compress(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device, dtype=torch.float32)
        # (N, In) @ (In, Out)^T → (N, Out)
        y = self._output_function(X @ self.W.T)
        return y.detach().cpu().numpy()

    def decompress(self, compressed_X):
        # compressed_X가 np.ndarray든 Tensor든 모두 처리
        if not isinstance(compressed_X, torch.Tensor):
            Y = torch.tensor(compressed_X, dtype=torch.float32, device=self.device)
        else:
            Y = compressed_X.to(self.device, dtype=torch.float32)
        # (N, Out) @ (Out, In) → (N, In)
        X_reconstructed = self._output_function(Y @ self.W)
        return X_reconstructed.detach().cpu().numpy()


class MultiBAMv3_Huber:
    """
    Multi-Layer Wrapper for BAMv3_Huber
    """
    def __init__(self, layers_dims, eta=1e-4, huber_delta=0.01, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bams = []
        
        for i in range(len(layers_dims) - 1):
            bam = BAMv3_Huber(
                input_dim=layers_dims[i], 
                output_dim=layers_dims[i + 1], 
                eta=eta, 
                huber_delta=huber_delta, 
                device=self.device
            )
            self.bams.append(bam)

    def train(self, X, num_epochs=1, batch_size=32):
        all_losses = []
        for i, bam in enumerate(self.bams):
            print(f"   > Layer {i+1} Training... (Delta: {bam.huber_delta})")
            losses = bam.train(X, num_epochs=num_epochs, batch_size=batch_size)
            all_losses.append(losses)
            X = bam.compress(X)  # 다음 레이어 입력
        return all_losses

    def compress(self, X):
        for bam in self.bams:
            X = bam.compress(X)
        return X

    def decompress(self, X):
        for bam in reversed(self.bams):
            X = bam.decompress(X)
        return X

  
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BAMv4(nn.Module):
    """
    BAMv4: Denoising + Compression용 단일 레이어 BAM
    - Encoder:  x_noisy  -> z  (dim: input_dim -> latent_dim)
    - Decoder:  z        -> x_hat (dim: latent_dim -> input_dim)
      (decoder는 encoder weight의 transpose 사용 = BAM flavor 유지)
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation: str = "tanh",   # "tanh" or "relu" or "none"
        device: str = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Encoder weight (latent_dim x input_dim)
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)

        if activation == "tanh":
            self.activation = torch.tanh

        elif activation == "relu":
            self.activation = F.relu

        elif activation == "leaky_relu":
            self.activation = F.leaky_relu

        elif activation is None or activation == "none":
            self.activation = None

        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.to(self.device)

    # --------- 기본 forward / encode / decode ---------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim)
        return: (B, latent_dim)
        """
        z = self.encoder(x)
        if self.activation is not None:
            z = self.activation(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        decoder는 encoder weight의 transpose 사용
        z: (B, latent_dim)
        return: (B, input_dim)
        """
        W = self.encoder.weight  # (latent_dim, input_dim)
        x_hat = torch.matmul(z, W)  # (B, input_dim)
        if self.activation is not None:
            x_hat = self.activation(x_hat)
        return x_hat

    def forward(self, x_noisy: torch.Tensor) -> torch.Tensor:
        """
        Denoising forward:
        x_noisy -> z -> x_hat (clean 쪽으로 복원)
        """
        z = self.encode(x_noisy)
        x_hat = self.decode(z)
        return x_hat

    # --------- Numpy <-> Tensor 헬퍼 ---------
    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        return x.to(self.device)

    def reconstruct_numpy(self, x_noisy_np: np.ndarray) -> np.ndarray:
        """
        Numpy 입력을 받아서 numpy 결과로 반환 (배치 형태)
        """
        self.eval()
        with torch.no_grad():
            x_noisy = self._to_tensor(x_noisy_np)
            x_hat = self.forward(x_noisy)
            return x_hat.cpu().numpy()

    def compress_numpy(self, x_np: np.ndarray) -> np.ndarray:
        """
        Numpy 입력 -> latent z (numpy)
        """
        self.eval()
        with torch.no_grad():
            x = self._to_tensor(x_np)
            z = self.encode(x)
            return z.cpu().numpy()

    def decompress_numpy(self, z_np: np.ndarray) -> np.ndarray:
        """
        latent z (numpy) -> 복원 x_hat (numpy)
        """
        self.eval()
        with torch.no_grad():
            z = self._to_tensor(z_np)
            x_hat = self.decode(z)
            return x_hat.cpu().numpy()

    # --------- Denoising 학습용 메서드 ---------
    def fit_denoise(
        self,
        X_noisy,
        X_clean,
        num_epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        verbose: bool = True
    ):
        """
        Denoising 학습:
        - 입력:  X_noisy  (노이즈 낀 데이터)   shape: (N, input_dim)
        - 타깃:  X_clean  (클린 데이터)       shape: (N, input_dim)
        """
        Xn = self._to_tensor(X_noisy)
        Xc = self._to_tensor(X_clean)
        assert Xn.shape == Xc.shape, "Noisy/Clean shape mismatch"

        dataset = torch.utils.data.TensorDataset(Xn, Xc)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        history = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            self.train()

            for batch_noisy, batch_clean in loader:
                optimizer.zero_grad()
                x_hat = self.forward(batch_noisy)          # noisy -> clean 추정
                loss = criterion(x_hat, batch_clean)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_noisy.size(0)

            epoch_loss /= len(dataset)
            history.append(epoch_loss)
            if verbose:
                print(f"[BAMv4] Epoch {epoch+1}/{num_epochs} | MSE = {epoch_loss:.6f}")

        return history


class MultiBAMv4(nn.Module):
    """
    MultiBAMv4: 여러 층을 쌓은 BAM식 Denoising Autoencoder
    - encoder_layers: Linear(input_dim -> h1 -> h2 -> ... -> hL)
    - decoder는 각 layer weight의 transpose를 역순으로 사용
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,       # 예: [2048, 512, 128]  (마지막 128이 compressed dim)
        activation: str = "tanh",
        device: str = None
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Encoder 레이어 정의
        dims = [input_dim] + hidden_dims
        self.encoder_layers = nn.ModuleList()
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layer = nn.Linear(in_d, out_d, bias=False)
            self.encoder_layers.append(layer)

        # Activation
        if activation == "tanh":
            self.activation = torch.tanh

        elif activation == "relu":
            self.activation = F.relu

        elif activation == "leaky_relu":
            self.activation = F.leaky_relu

        elif activation is None or activation == "none":
            self.activation = None

        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.to(self.device)

    # ---------- Forward helpers ----------
    def _act(self, x):
        return self.activation(x) if self.activation is not None else x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim)
        return: (B, last_hidden_dim)
        """
        h = x
        for layer in self.encoder_layers:
            h = self._act(layer(h))
        return h

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, last_hidden_dim)
        decoder는 encoder weight의 transpose를 역순으로 사용
        """
        h = z
        for layer in reversed(self.encoder_layers):
            W = layer.weight  # (out_dim, in_dim)
            h = torch.matmul(h, W)  # (B, in_dim)
            h = self._act(h)
        return h

    def forward(self, x_noisy: torch.Tensor) -> torch.Tensor:
        z = self.encode(x_noisy)
        x_hat = self.decode(z)
        return x_hat

    # ---------- Numpy helpers ----------
    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        return x.to(self.device)

    def reconstruct_numpy(self, x_noisy_np: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x_noisy = self._to_tensor(x_noisy_np)
            x_hat = self.forward(x_noisy)
            return x_hat.cpu().numpy()

    def compress_numpy(self, x_np: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = self._to_tensor(x_np)
            z = self.encode(x)
            return z.cpu().numpy()

    def decompress_numpy(self, z_np: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            z = self._to_tensor(z_np)
            x_hat = self.decode(z)
            return x_hat.cpu().numpy()

    # ---------- Denoising 학습 ----------
    def fit_denoise(
        self,
        X_noisy,
        X_clean,
        num_epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        verbose: bool = True
    ):
        """
        X_noisy: (N, input_dim)
        X_clean: (N, input_dim)
        """
        Xn = self._to_tensor(X_noisy)
        Xc = self._to_tensor(X_clean)
        assert Xn.shape == Xc.shape, "Noisy/Clean shape mismatch"

        dataset = torch.utils.data.TensorDataset(Xn, Xc)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        history = []

        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0

            for batch_noisy, batch_clean in loader:
                optimizer.zero_grad()
                x_hat = self.forward(batch_noisy)
                loss = criterion(x_hat, batch_clean)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_noisy.size(0)

            epoch_loss /= len(dataset)
            history.append(epoch_loss)

            if verbose:
                print(f"[MultiBAMv4] Epoch {epoch+1}/{num_epochs} | MSE = {epoch_loss:.6f}")

        return history
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBAMv5(nn.Module):
    """
    MultiBAMv5
    - Untied encoder/decoder
    - Residual output
    - Optional BatchNorm
    - Flexible activation functions
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        activation="relu",
        residual_alpha=1.0,
        use_bn=False,
        device=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.residual_alpha = residual_alpha
        self.use_bn = use_bn

        # device 설정
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # -----------------------------
        # Activation 설정
        # -----------------------------
        act = activation.lower() if isinstance(activation, str) else activation

        if act is None or act == "none":
            self.activation = None
        elif act == "tanh":
            self.activation = torch.tanh
        elif act == "relu":
            self.activation = F.relu
        elif act in ("leaky_relu", "leaky relu", "lrelu"):
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # -----------------------------
        # Encoder layers 생성
        # -----------------------------
        dims = [input_dim] + hidden_dims
        self.encoder_layers = nn.ModuleList()
        self.encoder_bns = nn.ModuleList()

        for in_d, out_d in zip(dims[:-1], dims[1:]):
            linear = nn.Linear(in_d, out_d, bias=False)
            nn.init.xavier_uniform_(linear.weight)
            self.encoder_layers.append(linear)

            if use_bn:
                self.encoder_bns.append(nn.BatchNorm1d(out_d))
            else:
                self.encoder_bns.append(None)

        # -----------------------------
        # Decoder layers 생성 (untied)
        # -----------------------------
        dec_dims = list(reversed(dims))
        self.decoder_layers = nn.ModuleList()
        self.decoder_bns = nn.ModuleList()

        for in_d, out_d in zip(dec_dims[:-1], dec_dims[1:]):
            linear = nn.Linear(in_d, out_d, bias=False)
            nn.init.xavier_uniform_(linear.weight)
            self.decoder_layers.append(linear)

            if use_bn:
                self.decoder_bns.append(nn.BatchNorm1d(out_d))
            else:
                self.decoder_bns.append(None)

        self.to(self.device)

    # -----------------------------
    # encoder
    # -----------------------------
    def encode(self, x):
        h = x
        for idx, layer in enumerate(self.encoder_layers):
            h = layer(h)
            if self.encoder_bns[idx] is not None:
                h = self.encoder_bns[idx](h)
            if self.activation is not None:
                h = self.activation(h)
        return h

    # -----------------------------
    # decoder
    # -----------------------------
    def decode(self, z):
        h = z
        for idx, layer in enumerate(self.decoder_layers):
            h = layer(h)
            if self.decoder_bns[idx] is not None:
                h = self.decoder_bns[idx](h)
            if self.activation is not None:
                h = self.activation(h)
        return h

    # -----------------------------
    # forward (Residual 포함)
    # -----------------------------
    def forward(self, x_noisy):
        z = self.encode(x_noisy)
        delta = self.decode(z)
        return x_noisy + self.residual_alpha * delta

    # -----------------------------
    # numpy 버전
    # -----------------------------
    def _to_tensor(self, x):
        if isinstance(x, (list, tuple)):
            x = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        return x.to(self.device)

    def reconstruct_numpy(self, x_noisy_np):
        self.eval()
        with torch.no_grad():
            x_noisy = self._to_tensor(x_noisy_np)
            x_hat = self.forward(x_noisy)
            return x_hat.cpu().numpy()

    # -----------------------------
    # 학습 함수
    # -----------------------------
    def fit_denoise(
        self,
        X_noisy,
        X_clean,
        num_epochs=1,
        batch_size=64,
        lr=1e-3,
        weight_decay=0.0,
        verbose=True,
    ):
        Xn = self._to_tensor(X_noisy)
        Xc = self._to_tensor(X_clean)

        dataset = torch.utils.data.TensorDataset(Xn, Xc)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.HuberLoss(delta=0.02)

        history = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for noisy, clean in loader:
                optimizer.zero_grad()
                out = self.forward(noisy)
                loss = criterion(out, clean)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * noisy.size(0)

            epoch_loss /= len(dataset)
            history.append(epoch_loss)

            if verbose:
                print(f"[v5] Epoch {epoch+1}/{num_epochs} | Loss = {epoch_loss:.6f}")

        return history


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualFCBlock(nn.Module):
    """
    Fully-connected residual block:
    y = skip(x) + main(x)
    - main: Linear -> (BN) -> act -> Linear
    - skip: identity or Linear (if in_dim != out_dim)
    """
    def __init__(self, in_dim, out_dim, activation="relu", use_bn=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bn = use_bn

        # activation 함수 선택
        act = activation.lower() if isinstance(activation, str) else activation
        if act is None or act == "none":
            self.act_fn = None
        elif act == "tanh":
            self.act_fn = torch.tanh
        elif act == "relu":
            self.act_fn = F.relu
        elif act in ("leaky_relu", "leaky relu", "lrelu"):
            self.act_fn = F.leaky_relu
        else:
            raise ValueError(f"Unknown activation in ResidualFCBlock: {activation}")

        # main branch
        hidden_dim = max(out_dim, in_dim)  # 중간 차원, 너무 작게 줄이지 않게
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        if use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(out_dim)
        else:
            self.bn1 = None
            self.bn2 = None

        # skip branch (projection if needed)
        if in_dim != out_dim:
            self.skip_proj = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.xavier_uniform_(self.skip_proj.weight)
        else:
            self.skip_proj = None

    def forward(self, x):
        # main
        h = self.fc1(x)
        if self.bn1 is not None:
            h = self.bn1(h)
        if self.act_fn is not None:
            h = self.act_fn(h)

        h = self.fc2(h)
        if self.bn2 is not None:
            h = self.bn2(h)

        # skip
        if self.skip_proj is not None:
            s = self.skip_proj(x)
        else:
            s = x

        out = h + s
        # 마지막에 activation 한 번 더 줄지 말지는 선택인데, 여기서는 주는 쪽으로
        if self.act_fn is not None:
            out = self.act_fn(out)
        return out


class MultiBAMv6(nn.Module):
    """
    MultiBAMv6
    - Encoder/Decoder: Residual FC Blocks (untied)
    - Outer residual: x_hat = x_noisy + alpha * delta
    - Optional BatchNorm inside blocks
    """
    def __init__(
        self,
        input_dim,
        hidden_dims,
        activation="relu",
        residual_alpha=1.0,
        use_bn=False,
        device=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.residual_alpha = residual_alpha
        self.use_bn = use_bn

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # encoder: input_dim -> hidden_dims[-1]
        dims = [input_dim] + hidden_dims
        self.encoder_blocks = nn.ModuleList()
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            block = ResidualFCBlock(in_d, out_d, activation=activation, use_bn=use_bn)
            self.encoder_blocks.append(block)

        # decoder: reverse path
        dec_dims = list(reversed(dims))
        self.decoder_blocks = nn.ModuleList()
        for in_d, out_d in zip(dec_dims[:-1], dec_dims[1:]):
            block = ResidualFCBlock(in_d, out_d, activation=activation, use_bn=use_bn)
            self.decoder_blocks.append(block)
            
        self.out_linear = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.xavier_uniform_(self.out_linear.weight)

        

        self.to(self.device)

    # ---- helper ----
    def _to_tensor(self, x):
        if isinstance(x, (list, tuple)):
            x = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        return x.to(self.device)

    # ---- encode/decode ----
    def encode(self, x):
        h = x
        for block in self.encoder_blocks:
            h = block(h)
        return h

    def decode(self, z):
        h = z
        for block in self.decoder_blocks:
            h = block(h)
        h = self.out_linear(h)     # ✅ 여기로 옮겨도 됨
        return h

    # ---- forward (outer residual) ----
    def forward(self, x_noisy):
        z = self.encode(x_noisy)
        delta = self.decode(z)
        return x_noisy + self.residual_alpha * delta

    # ---- numpy interface ----
    def reconstruct_numpy(self, x_noisy_np):
        self.eval()
        with torch.no_grad():
            x_noisy = self._to_tensor(x_noisy_np)
            x_hat = self.forward(x_noisy)
            return x_hat.cpu().numpy()

    # ---- training ----
    def fit_denoise(
        self,
        X_noisy,
        X_clean,
        num_epochs=1,
        batch_size=64,
        lr=1e-4,
        weight_decay=0.0,
        verbose=True,
    ):
        Xn = self._to_tensor(X_noisy)
        Xc = self._to_tensor(X_clean)

        dataset = torch.utils.data.TensorDataset(Xn, Xc)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        history = []
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0
            for noisy, clean in loader:
                optimizer.zero_grad()
                out = self.forward(noisy)
                loss = criterion(out, clean)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item() * noisy.size(0)

            epoch_loss /= len(dataset)
            history.append(epoch_loss)
            if verbose:
                print(f"[v6] Epoch {epoch+1}/{num_epochs} | Loss = {epoch_loss:.6f}")

        return history

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexIQBAM(nn.Module):
    """
    Single-layer Complex-Valued BAM (paper-style dynamics) for IQ denoising.

    - Input/Output: complex IQ sequence
      * accepts [B, T] or [B, T, 1] (torch.cfloat)
      * returns same shape as input

        - Dynamics (Euler, inner steps):
                dv/dt = -Dv * v + g * W_uv * f(u)
                du/dt = -Du * u + g * W_vu * f(v) + (inject_input ? x_t : 0)

            여기서 `inject_input`은 **inner step 동안 외부 입력(x_t)을 계속 주입할지**를 제어합니다.
            - inject_input=True  : 논문식 forcing 형태에 가깝게, 매 inner step에서 +x_t를 포함
            - inject_input=False : inner step은 자율 동역학(autonomous)으로 수렴(단, u 초기화/anchor는 별도)

    - Activation: split tanh (tanh on real/imag separately), as in common complex BAM assumptions.
    - Decay: positive via softplus + floor.
    - Optional stability: gain clamped by Du,Dv and weight norms (can be turned off).
    """

    def __init__(
        self,
        m_units: int,               # size of V layer (memory)
        dt: float = 0.05,
        steps: int = 15,
        decay_init: float = 1.0,    # initial decay (positive after softplus)
        decay_floor: float = 0.1,   # minimal decay to avoid near-zero
        w_gain_init: float = 0.1,   # coupling gain init
        stability_margin: float = 0.5,  # <1 means stricter stability
        use_gain_clamp: bool = True,
        anchor_strength: float = 0.10,  # how strongly to keep u close to input (prevents drift)
        inject_input: bool = True,      # whether to inject x_t at every inner step
        init_scale: float = 0.01,       # complex weight init scale
    ):
        super().__init__()
        self.n = 1
        self.m = int(m_units)
        self.dt = float(dt)
        self.steps = int(steps)
        self.decay_floor = float(decay_floor)
        self.stability_margin = float(stability_margin)
        self.use_gain_clamp = bool(use_gain_clamp)
        self.anchor_strength = float(anchor_strength)
        self.inject_input = bool(inject_input)

        # Positive decays (learnable)
        self.raw_decay_u = nn.Parameter(torch.ones(self.n) * float(decay_init))
        self.raw_decay_v = nn.Parameter(torch.ones(self.m) * float(decay_init))

        # Gain (learnable)
        self.raw_w_gain = nn.Parameter(torch.tensor(float(w_gain_init)))

        # Complex weights: U(1) <-> V(m)
        self.W_uv = nn.Linear(self.n, self.m, bias=False, dtype=torch.cfloat)  # u -> v
        self.W_vu = nn.Linear(self.m, self.n, bias=False, dtype=torch.cfloat)  # v -> u

        self._init_complex_weights(init_scale)

    # ------------------------
    # init / constraints
    # ------------------------
    def _init_complex_weights(self, init_scale: float):
        with torch.no_grad():
            su = init_scale / math.sqrt(max(1, self.n))
            sv = init_scale / math.sqrt(max(1, self.m))
            self.W_uv.weight.real.normal_(0, su)
            self.W_uv.weight.imag.normal_(0, su)
            self.W_vu.weight.real.normal_(0, sv)
            self.W_vu.weight.imag.normal_(0, sv)

    def decay_u(self):  
        return F.softplus(self.raw_decay_u) + self.decay_floor  # shape (1,)

    def decay_v(self):
        return F.softplus(self.raw_decay_v) + self.decay_floor  # shape (m,)

    def w_gain(self):
        g = F.softplus(self.raw_w_gain)

        if not self.use_gain_clamp:
            return g

        # Simple norm-based clamp:
        # g^2 * ||W_uv|| * ||W_vu|| < min(Du) * min(Dv) * margin
        Du_min = self.decay_u().min()
        Dv_min = self.decay_v().min()

        # NOTE: computed without grad (hard constraint)
        with torch.no_grad():
            norm_uv = torch.linalg.norm(self.W_uv.weight)
            norm_vu = torch.linalg.norm(self.W_vu.weight)

        max_g2 = (Du_min * Dv_min * self.stability_margin) / (norm_uv * norm_vu + 1e-8)
        max_g = torch.sqrt(torch.clamp(max_g2, min=0.0))
        return torch.minimum(g, max_g)

    # ------------------------
    # activation (paper assumption)
    # ------------------------
    @staticmethod
    def split_tanh(z: torch.Tensor) -> torch.Tensor:
        return torch.complex(torch.tanh(z.real), torch.tanh(z.imag))

    # ------------------------
    # one dynamics step
    # ------------------------
    def dynamics_step(self, u: torch.Tensor, v: torch.Tensor, x_t: torch.Tensor):
        """
        u: [B,1] complex
        v: [B,m] complex
        x_t: [B,1] complex (external input at time t)
        """
        Du = self.decay_u()  # [1]
        Dv = self.decay_v()  # [m]
        g = self.w_gain()    # scalar

        fu = self.split_tanh(u)
        fv = self.split_tanh(v)

        du = -Du * u + g * self.W_vu(fv)
        if self.inject_input:
            du = du + x_t
        dv = -Dv * v + g * self.W_uv(fu)

        u_new = u + self.dt * du
        v_new = v + self.dt * dv
        return u_new, v_new

    # ------------------------
    # forward
    # ------------------------
    def forward(
        self,
        noisy_iq: torch.Tensor,
        v0: torch.Tensor | None = None,
        *,
        return_trace: bool = False,
        trace_t: int | None = None,
    ):
        """
        noisy_iq: [B,T] or [B,T,1] complex (torch.cfloat)
        returns : same shape

        return_trace=True 이면 (out, trace)를 반환합니다.
        - trace_t: 추적할 시간 인덱스 t (None이면 가운데 t=T//2)
        - trace에는 해당 t에서 inner step별 u의 변화 및 노름/오차 요약이 담깁니다.
        """
        if noisy_iq.dtype != torch.cfloat and noisy_iq.dtype != torch.cdouble:
            raise TypeError("noisy_iq must be complex (torch.cfloat or torch.cdouble).")

        squeeze_last = False
        if noisy_iq.dim() == 2:
            # [B,T] -> [B,T,1]
            noisy_iq = noisy_iq.unsqueeze(-1)
            squeeze_last = True
        elif noisy_iq.dim() == 3 and noisy_iq.size(-1) == 1:
            pass
        else:
            raise ValueError("Expected noisy_iq shape [B,T] or [B,T,1].")

        B, T, _ = noisy_iq.shape
        device = noisy_iq.device
        dtype = noisy_iq.dtype

        u = torch.zeros(B, 1, dtype=dtype, device=device)
        v = torch.zeros(B, self.m, dtype=dtype, device=device) if v0 is None else v0.to(device=device, dtype=dtype)

        outs = []

        do_trace = bool(return_trace)
        if do_trace:
            if trace_t is None:
                trace_t = int(T // 2)
            trace_t = int(max(0, min(int(trace_t), T - 1)))

            # store only what we need for logging (avoid huge memory)
            trace_u = []        # list of [B,1]
            trace_u_l2 = []     # list of scalar tensors
            trace_v_l2 = []     # list of scalar tensors
            trace_ux_l2 = []    # ||u-x_t||^2 mean
        for t in range(T):
            x_t = noisy_iq[:, t, :]  # [B,1]

            # initialize u near current input (helps convergence early)
            u = x_t

            if do_trace and t == trace_t:
                trace_u.append(u)
                trace_u_l2.append((u.real**2 + u.imag**2).mean())
                trace_v_l2.append((v.real**2 + v.imag**2).mean())
                trace_ux_l2.append(((u - x_t).real**2 + (u - x_t).imag**2).mean())

            # inner convergence loop
            for s in range(self.steps):
                u, v = self.dynamics_step(u, v, x_t)

                # anchor to input to prevent drift / preserve information
                if self.anchor_strength > 0:
                    # optionally decay anchor during steps
                    a = self.anchor_strength * (1.0 - (s / max(1, self.steps)))
                    u = (1 - a) * u + a * x_t

                if do_trace and t == trace_t:
                    trace_u.append(u)
                    trace_u_l2.append((u.real**2 + u.imag**2).mean())
                    trace_v_l2.append((v.real**2 + v.imag**2).mean())
                    trace_ux_l2.append(((u - x_t).real**2 + (u - x_t).imag**2).mean())

            outs.append(u)

        out = torch.stack(outs, dim=1)  # [B,T,1]
        out = out.squeeze(-1) if squeeze_last else out

        if not do_trace:
            return out

        trace = {
            "trace_t": int(trace_t),
            "steps": int(self.steps),
            "dt": float(self.dt),
            "inject_input": bool(self.inject_input),
            "anchor_strength": float(self.anchor_strength),
            "u": torch.stack(trace_u, dim=0),              # [S+1, B, 1]
            "u_l2": torch.stack(trace_u_l2, dim=0),        # [S+1]
            "v_l2": torch.stack(trace_v_l2, dim=0),        # [S+1]
            "ux_l2": torch.stack(trace_ux_l2, dim=0),      # [S+1]
        }
        return out, trace

    @torch.no_grad()
    def stability_info(self):
        Du_min = self.decay_u().min().item()
        Dv_min = self.decay_v().min().item()
        g = self.w_gain().item()
        norm_uv = torch.linalg.norm(self.W_uv.weight).item()
        norm_vu = torch.linalg.norm(self.W_vu.weight).item()
        lhs = (g ** 2) * norm_uv * norm_vu
        rhs = Du_min * Dv_min
        return {
            "is_stable_like": lhs < rhs,
            "stability_ratio(lhs/rhs)": lhs / (rhs + 1e-12),
            "Du_min": Du_min,
            "Dv_min": Dv_min,
            "g": g,
            "||W_uv||": norm_uv,
            "||W_vu||": norm_vu,
        }
# # --------------------------
# # U-NET (small, configurable)
# # --------------------------
# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x):
#         return self.conv(x)

# class UNet2Ch(nn.Module):
#     def __init__(self, in_ch=2, base_filters=32):
#         super().__init__()
#         f = base_filters
#         # encoder
#         self.c1 = ConvBlock(in_ch, f)
#         self.p1 = nn.MaxPool2d(2)
#         self.c2 = ConvBlock(f, f*2)
#         self.p2 = nn.MaxPool2d(2)
#         self.c3 = ConvBlock(f*2, f*4)
#         self.p3 = nn.MaxPool2d(2)
#         # bottleneck
#         self.b = ConvBlock(f*4, f*8)
#         # decoder
#         self.u3 = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)
#         self.c4 = ConvBlock(f*8, f*4)
#         self.u2 = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)
#         self.c5 = ConvBlock(f*4, f*2)
#         self.u1 = nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)
#         self.c6 = ConvBlock(f*2, f)
#         # final
#         self.out_conv = nn.Conv2d(f, 2, kernel_size=1)  # 2 channels: Re_hat, Im_hat

#     def forward(self, x):
#         # x: (B, 2, F, T)
#         c1 = self.c1(x)
#         p1 = self.p1(c1)
#         c2 = self.c2(p1)
#         p2 = self.p2(c2)
#         c3 = self.c3(p2)
#         p3 = self.p3(c3)

#         b = self.b(p3)

#         u3 = self.u3(b)
#         u3 = torch.cat([u3, c3], dim=1)
#         c4 = self.c4(u3)

#         u2 = self.u2(c4)
#         u2 = torch.cat([u2, c2], dim=1)
#         c5 = self.c5(u2)

#         u1 = self.u1(c5)
#         u1 = torch.cat([u1, c1], dim=1)
#         c6 = self.c6(u1)

#         out = self.out_conv(c6)  # linear outputs (can be negative)
#         return out

# # --------------------------
# # Dataset helpers
# # --------------------------
# class SpectrogramDataset(Dataset):
#     """
#     expects inputs:
#       - spec_re: numpy array (N, F, T)
#       - spec_im: numpy array (N, F, T)
#     Returns torch tensors normalized per-sample by max magnitude.
#     """
#     def __init__(self, spec_re, spec_im, normalize=True):
#         assert spec_re.shape == spec_im.shape
#         self.re = spec_re.astype(np.float32)
#         self.im = spec_im.astype(np.float32)
#         self.normalize = normalize

#     def __len__(self):
#         return self.re.shape[0]

#     def __getitem__(self, idx):
#         Re = self.re[idx]
#         Im = self.im[idx]
#         mag = np.sqrt(Re**2 + Im**2)
#         max_val = mag.max() if self.normalize else 1.0
#         if max_val == 0:
#             max_val = 1.0
#         Re_n = Re / max_val
#         Im_n = Im / max_val
#         # return stacked (2, F, T) and the scale factor for ISTFT inversion
#         return torch.from_numpy(np.stack([Re_n, Im_n], axis=0)), float(max_val)

# # --------------------------
# # Loss helpers (mag + time-domain)
# # --------------------------
# def magnitude_mse_loss(re_true, im_true, re_hat, im_hat):
#     mag = torch.sqrt(re_true**2 + im_true**2 + 1e-12)
#     mag_hat = torch.sqrt(re_hat**2 + im_hat**2 + 1e-12)
#     return F.mse_loss(mag_hat, mag)

# def waveform_l1_loss_from_reim(re_hat, im_hat, re_true, im_true, istft_params):
#     """
#     Re/Im all tensors shaped (B, F, T)
#     istft_params: dict with n_fft, hop_length, win_length, window (torch.Tensor or None)
#     Returns L1 between original time waveform and predicted waveform
#     NOTE: This expects you have the original time waveform to compare against.
#     If you don't have it, you can compare reconstructed waveform from ground-truth spectrogram.
#     """
#     # Build complex tensors
#     complex_hat = torch.complex(re_hat, im_hat)   # shape (B, F, T)
#     complex_true = torch.complex(re_true, im_true)
#     # inverse STFT: torch.istft expects (..., freq, frames)
#     x_hat = torch.istft(complex_hat, **istft_params)
#     x_true = torch.istft(complex_true, **istft_params)
#     return F.l1_loss(x_hat, x_true)

# # --------------------------
# # Training loop
# # --------------------------
# def train_unet(
#     model, dataloader, istft_params,
#     device='cuda' if torch.cuda.is_available() else 'cpu',
#     lr=1e-3, n_epochs=50, alpha=1.0, beta=0.5
# ):
#     model = model.to(device)
#     opt = torch.optim.Adam(model.parameters(), lr=lr)
#     for epoch in range(1, n_epochs+1):
#         model.train()
#         running_loss = 0.0
#         for batch_idx, (x, scales) in enumerate(dataloader):
#             # x: (B, 2, F, T)
#             x = x.to(device)      # normalized Re/Im
#             scales = torch.tensor(scales, device=device).float()

#             # Forward
#             pred = model(x)       # (B, 2, F, T)
#             re_true = x[:,0]
#             im_true = x[:,1]
#             re_hat = pred[:,0]
#             im_hat = pred[:,1]

#             # Undo normalization per-sample before waveform ISTFT if needed:
#             # shape: (B, F, T)
#             re_true_scaled = re_true * scales.view(-1,1,1)
#             im_true_scaled = im_true * scales.view(-1,1,1)
#             re_hat_scaled = re_hat * scales.view(-1,1,1)
#             im_hat_scaled = im_hat * scales.view(-1,1,1)

#             # Loss terms
#             L_mag = magnitude_mse_loss(re_true, im_true, re_hat, im_hat)  # on normalized spectrograms
#             # Wave L1 (compute with scaled re/im)
#             L_wave = waveform_l1_loss_from_reim(
#                 re_hat_scaled, im_hat_scaled, re_true_scaled, im_true_scaled, istft_params
#             )

#             loss = alpha * L_mag + beta * L_wave

#             opt.zero_grad()
#             loss.backward()
#             opt.step()

#             running_loss += loss.item()

#         avg = running_loss / len(dataloader)
#         print(f"Epoch {epoch}/{n_epochs} - Loss: {avg:.6f}")

# # --------------------------
# # Usage example (synth / adapt to your data)
# # --------------------------
# if __name__ == "__main__":
    # === Dummy / example shapes ===
    # Suppose your spectrograms have F=256 frequency bins, T=64 time frames
    # N = 200   # number of samples
    # F = 256
    # T = 64

    # # Replace these with your real spectrogram arrays (N, F, T)
    # # Here we synthesize some example complex spectrograms:
    # rng = np.random.RandomState(0)
    # spec_re = rng.normal(scale=0.1, size=(N, F, T)).astype(np.float32)
    # spec_im = rng.normal(scale=0.1, size=(N, F, T)).astype(np.float32)

    # dataset = SpectrogramDataset(spec_re, spec_im, normalize=True)
    # loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    # # ISTFT parameters - must match the STFT used to create your spectrograms
    # n_fft = (F - 1) * 2  # e.g., F = n_fft//2 + 1
    # hop_length = n_fft // 4
    # win_length = n_fft
    # window = torch.hann_window(win_length)

    # istft_params = dict(n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True, normalized=False, onesided=True)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = UNet2Ch(in_ch=2, base_filters=32)
    # train_unet(model, loader, istft_params, device=device, lr=1e-3, n_epochs=30, alpha=1.0, beta=0.3)

    # # After training: inference example
    # model.eval()
    # sample, scale = dataset[0]
    # with torch.no_grad():
    #     x = sample.unsqueeze(0).to(device)  # (1,2,F,T)
    #     pred = model(x).cpu().numpy()[0]    # (2,F,T)
    # re_hat = pred[0] * scale
    # im_hat = pred[1] * scale
    # # re_hat/im_hat are your reconstructed spectrogram channels

