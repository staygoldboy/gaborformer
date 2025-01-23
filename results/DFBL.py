import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal as signal
import matplotlib.pyplot as plt
import librosa




# 1. Gabor Filter (Time-Domain Filtering)
class GaborFilter(nn.Module):
    def __init__(self, num_filters, filter_length, sample_rate):
        super(GaborFilter, self).__init__()
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.sample_rate = sample_rate

        # Define learnable parameters: center frequency and bandwidth
        self.center_freq = nn.Parameter(torch.rand(num_filters) * sample_rate / 2)
        self.bandwidth = nn.Parameter(torch.rand(num_filters) * sample_rate / 10)

        # Time axis for the filter
        self.t = torch.linspace(-filter_length // 2, filter_length // 2, filter_length).to(next(self.parameters()).device) / sample_rate
    def forward(self, x):
        # Generate Gabor filters (using only real part and imaginary part)
        filters_real = []
        filters_imag = []
        for i in range(self.num_filters):
            gabor_real = torch.exp(
                -0.5 * (self.t ** 2).to(x.device) / (self.bandwidth[i] ** 2).to(x.device)) * torch.cos(
                2 * np.pi * self.center_freq[i] * self.t.to(x.device)
            )
            gabor_imag = torch.exp(
                -0.5 * (self.t ** 2).to(x.device) / (self.bandwidth[i] ** 2).to(x.device)) * torch.sin(
                2 * np.pi * self.center_freq[i] * self.t.to(x.device)
            )

            filters_real.append(gabor_real)
            filters_imag.append(gabor_imag)

        filters_real = torch.stack(filters_real).to(x.device)
        filters_imag = torch.stack(filters_imag).to(x.device)

        # Convolve input with Gabor filters (time-domain filtering)
        real_output = F.conv1d(x, filters_real.view(self.num_filters, 1, -1), padding=self.filter_length // 2)
        imag_output = F.conv1d(x, filters_imag.view(self.num_filters, 1, -1), padding=self.filter_length // 2)

        # Compute energy (sum of squares of real and imaginary parts)
        filtered_signal = torch.sqrt(real_output ** 2 + imag_output ** 2)

        return filtered_signal


# 2. Adaptive Scalogram Pooling
class AdaptivePooling(nn.Module):
    def __init__(self, output_size):
        super(AdaptivePooling, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        # Downsample the feature maps using average pooling
        pooled_signal = F.adaptive_avg_pool1d(x, self.output_size)
        return pooled_signal


# 3. Parameterized Nonlinear Transformation
class NonlinearTransformation(nn.Module):
    def __init__(self, num_channels):
        super(NonlinearTransformation, self).__init__()
        # Learnable parameters for nonlinear transformation
        self.offsets = nn.Parameter(torch.rand(num_channels))
        self.exponents = nn.Parameter(torch.rand(num_channels))

    def forward(self, x):
        # Apply nonlinear transformation to compress the dynamic range
        transformed_signal = (x + self.offsets.unsqueeze(1)) ** self.exponents.unsqueeze(1)
        return transformed_signal


# Full DFBL Module
class DFBL(nn.Module):
    def __init__(self, num_filters=64, filter_length=401, sample_rate=16000, output_size=128):
        super(DFBL, self).__init__()
        self.gabor_filter = GaborFilter(num_filters, filter_length, sample_rate)
        self.pooling = AdaptivePooling(output_size)
        self.nonlinear_transform = NonlinearTransformation(num_filters)

    def forward(self, x):
        # Step 1: Time-domain filtering
        filtered_signal = self.gabor_filter(x)

        # Step 2: Adaptive pooling
        pooled_signal = self.pooling(filtered_signal)

        # Step 3: Nonlinear transformation
        transformed_signal = self.nonlinear_transform(pooled_signal)

        return transformed_signal



# Example usage
if __name__ == '__main__':
    np.random.seed(0)
    data = np.load("MODMA_features_labels_selected.npz")
    feature = data['features'][0]
    labels = data['labels']
    feature1 = torch.tensor(feature).unsqueeze(0).unsqueeze(0)  # 扩展维度 (batch_size=1, channels=1)
    dfbl = DFBL()
    output = dfbl(feature1)
    print("input shape:", feature1.shape)
    print("Output shape:", output.shape)
    plt.imshow(output.squeeze(0).detach().numpy(), interpolation='nearest', cmap='hot')
    plt.show()

    # file_path=['/home/test/conformer/audio_lanzhou_2015/02030005/01.wav','/home/test/conformer/audio_lanzhou_2015/02030001/01.wav','/home/test/conformer/audio_lanzhou_2015/02010013/01.wav','/home/test/conformer/audio_lanzhou_2015/02010002/01.wav']
    # label=['HC','mild','moderate','severe']
    # sample_rate = 16000  # 采样率为16kHz
    # audio_signal=[]
    # for i in range(len(file_path)):
    #     audio_signal.append(librosa.load(file_path[i], sr=sample_rate)[0])
    # dfbl = DFBL()
    # fig, axs = plt.subplots(1, len(audio_signal), figsize=(15, 5))
    # for i in range(len(audio_signal)):
    #     audio_signal[i]=torch.tensor(audio_signal[i]).unsqueeze(0).unsqueeze(0)  # 扩展维度 (batch_size=1, channels=1)
    #     output = dfbl(audio_signal[i])
    #     print("input shape:", audio_signal[i].shape)
    #     print("Output shape:", output.shape)
    #     axs[i].imshow(output.squeeze(0).detach().numpy(), interpolation='nearest', cmap='hot')
    #     axs[i].set_title(label[i])
    #
    # plt.tight_layout()
    # plt.show()