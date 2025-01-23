import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample

# 加载音频文件
audio_file = 'negative_out.wav'
y, sr = librosa.load(audio_file, sr=None)

# 短期能量计算
frame_length = int(0.025 * sr)  # 25ms frame
hop_length = int(0.01 * sr)     # 10ms step
energy = np.array([
    np.sum(np.abs(y[i:i+frame_length])**2)
    for i in range(0, len(y), hop_length)
])

# 过零率计算
zero_crossings = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]

# 设置双阈值
high_threshold_energy = np.percentile(energy, 90)  # 高阈值，取能量90%的点
low_threshold_energy = np.percentile(energy, 10)   # 低阈值，取能量10%的点

# 初始化检测
start, end = None, None
voice_segments = []

# 双阈值检测
for i in range(len(energy)):
    if energy[i] > high_threshold_energy and start is None:
        # 检测到语音开始
        start = i
    elif energy[i] < low_threshold_energy and start is not None:
        # 检测到语音结束
        end = i
        # 检查语音片段是否有效（避免过短的片段）
        if (end - start) * hop_length >= 0.2 * sr:  # 保证至少200ms的语音段
            voice_segments.append((start, end))
        start = None

# 提取所有语音段并拼接
voice_audio = np.concatenate([y[start*hop_length:end*hop_length] for start, end in voice_segments])

# 重采样到16kHz
target_sr = 16000
if sr != target_sr:
    voice_audio_resampled = resample(voice_audio, int(len(voice_audio) * target_sr / sr))
else:
    voice_audio_resampled = voice_audio

# 保存处理后的人声部分
sf.write('clean_voice.wav', voice_audio_resampled, target_sr)

print("处理完成，输出文件: clean_voice.wav")