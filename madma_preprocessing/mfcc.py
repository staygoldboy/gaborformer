import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = './results/modma/mel_spectrogram'
# 文件路径
wav_files = {
    "hc": "/home/zlh/gabor/audio_lanzhou_16khz/02030002/23.wav",
    "mild": "/home/zlh/gabor/audio_lanzhou_16khz/02030008/10.wav",
    "moderate": "/home/zlh/gabor/audio_lanzhou_16khz/02010023/19.wav",
    "severe": "/home/zlh/gabor/audio_lanzhou_16khz/02010002/01.wav"
}

# 保存图像的函数
def save_mel_spectrogram(wav_path, output_path,label):

    # 加载音频文件
    y, sr = librosa.load(wav_path, sr=None)
    # 1. 预加重
    pre_emphasis = 0.97
    y_preemphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    # 计算 Mel 频谱
    S = librosa.feature.melspectrogram(y=y_preemphasized, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 绘制 Mel 频谱图
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
    # plt.colorbar(format='%+2.0f dB')
    plt.title(f'{label} Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, output_path), bbox_inches='tight', dpi=300)
    plt.close()

os.makedirs(save_dir, exist_ok=True)
# 提取 MFCC 并保存
for label, wav_path in wav_files.items():
    output_path = f"{label}_mel_spectrogram.png"
    save_mel_spectrogram(wav_path, output_path,label)




