import wave
import numpy as np
from pydub import AudioSegment,silence
import scipy.io.wavfile as wav
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import librosa
import opensmile
from scipy.signal import resample


def create_folders(root_dir):
    folders = ['clipped_data']
    subfolders = ['no_gender_balance']
    subsubfolders = {'audio': ['origin', 'selected']}

    os.makedirs(root_dir, exist_ok=True)
    for i in folders:
        for j in subfolders:
            for k, v in subsubfolders.items():
                for m in v:
                    # print(os.path.join(root_dir, i, j, k, m))
                    os.makedirs(os.path.join(root_dir, i, j, k, m), exist_ok=True)

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)

    return (data - mean) / std

def load_audio(audio_path):
    wavefile = wave.open(audio_path)
    audio_sr = wavefile.getframerate()
    n_samples = wavefile.getnframes()
    signal = np.frombuffer(wavefile.readframes(n_samples), dtype=np.short)

    return signal.astype(float), audio_sr

def audio_clipping(audio, audio_sr, text_df, zero_padding=False):
    if zero_padding:
        edited_audio = np.zeros(audio.shape[0])
        for t in text_df.itertuples():
            if getattr(t, 'speaker') == 'Participant':
                if 'scrubbed_entry' in getattr(t, 'value'):
                    continue
                else:
                    start = getattr(t, 'start_time')
                    stop = getattr(t, 'stop_time')
                    start_sample = int(start * audio_sr)
                    stop_sample = int(stop * audio_sr)
                    edited_audio[start_sample:stop_sample] = audio[start_sample:stop_sample]

        # cut head and tail of interview
        first_start = text_df['start_time'][0]
        last_stop = text_df['stop_time'][len(text_df) - 1]
        edited_audio = edited_audio[int(first_start * audio_sr):int(last_stop * audio_sr)]

    else:
        edited_audio = []
        for t in text_df.itertuples():
            if getattr(t, 'speaker') == 'Participant':
                if 'scrubbed_entry' in getattr(t, 'value'):
                    continue
                else:
                    start = getattr(t, 'start_time')
                    stop = getattr(t, 'stop_time')
                    start_sample = int(start * audio_sr)
                    stop_sample = int(stop * audio_sr)
                    edited_audio = np.hstack((edited_audio, audio[start_sample:stop_sample]))

    return edited_audio

def VAD(y,sr):
    # 短期能量计算
    frame_length = int(0.025 * sr)  # 25ms frame
    hop_length = int(0.01 * sr)  # 10ms step
    energy = np.array([
        np.sum(np.abs(y[i:i + frame_length]) ** 2)
        for i in range(0, len(y), hop_length)
    ])

    # 过零率计算
    zero_crossings = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]

    # 设置双阈值
    high_threshold_energy = np.percentile(energy, 90)  # 高阈值，取能量90%的点
    low_threshold_energy = np.percentile(energy, 10)  # 低阈值，取能量10%的点

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
    voice_audio = np.concatenate([y[start * hop_length:end * hop_length] for start, end in voice_segments])

    # 重采样到16kHz
    target_sr = 16000
    if sr != target_sr:
        voice_audio_resampled = resample(voice_audio, int(len(voice_audio) * target_sr / sr))
    else:
        voice_audio_resampled = voice_audio

    return voice_audio_resampled

def remove_silence(audio, sr, threshold_db=-40, min_silence_len=500,keep_silence=200):
    """
    删除音频中的静音片段

    参数:
    audio: 音频数据
    sr: 采样率
    threshold_db: 静音判定阈值(dB)
    min_silence_len: 最小静音长度(ms)

    返回:
    去除静音后的音频数据
    """
    # 将numpy数组转换为16位整数格式
    audio_int16 = (audio * 32767).astype(np.int16)

    # 创建正确的AudioSegment对象
    audio_segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,  # 固定使用16位(2字节)采样宽度
        channels=1
    )

    # 分割非静音片段
    chunks = silence.split_on_silence(
        audio_segment,
        min_silence_len=min_silence_len,
        silence_thresh=threshold_db,
        keep_silence=keep_silence
    )

    # 处理没有检测到静音片段的情况
    if not chunks:
        return audio

    # 合并非静音片段
    combined = chunks[0]
    for chunk in chunks[1:]:
        combined += chunk

    # 转回numpy数组并归一化
    combined_array = np.array(combined.get_array_of_samples(), dtype=np.float32)
    combined_array = combined_array / 32767.0  # 转回浮点数格式

    return combined_array

def create_sliding_windows(audio, sr, window_size=4, stride=1):
    """
    使用滑动窗口分割音频

    参数:
    audio: 音频数据
    sr: 采样率
    window_size: 窗口大小(秒)
    stride: 步幅(秒)

    返回:
    分割后的音频片段列表
    """
    # 转换为采样点数
    window_samples = int(window_size * sr)
    stride_samples = int(stride * sr)

    # 计算窗口数量
    if stride == 0:
        if len(audio) % window_samples ==0:
            n_windows = len(audio)//window_samples
        else:
            n_windows = (len(audio)//window_samples) + 1
    else:
        if (len(audio)-window_samples) % stride_samples ==0:
            n_windows = ((len(audio) - window_samples) // stride_samples) + 1
        else:
            n_windows = ((len(audio) - window_samples) // stride_samples) + 2

    # 创建窗口
    windows = []
    for i in range(n_windows):
        if stride==0:
            start = i * window_samples
            end = start + window_samples
        else:
            start = i * stride_samples
            end = start + window_samples

        window = audio[start:end]

        # 如果窗口长度不足，进行零填充
        if len(window) < window_samples:
            window = np.pad(window, (0, window_samples - len(window)))

        windows.append(window)

    return windows


def extract_feature(windows):
    # 初始化 openSMILE 并选择 emobase 特征集
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.emobase,
        feature_level=opensmile.FeatureLevel.Functionals
    )

    # 存储特征和标签的列表
    features = []
    labels = []

    for window in windows:
        # 提取特征
        feature_vector = smile.process_signal(window, sampling_rate=16000)
        features.append(feature_vector.values.flatten())

    # 将特征和标签转换为 NumPy 数组
    features = np.array(features)

    return features

def process_audio_file(audio_path, text_df,output_dir,ID,window_size=4, stride=1):
    """
    处理单个音频文件的完整流程
    """
    # 1. 移除采访者声音
    audio, sr = load_audio(audio_path)
    patient_audio= audio_clipping(audio,sr,text_df,zero_padding=False)

    # 2. 删除静音
    # no_silence_audio = remove_silence(patient_audio, sr)
    no_silence_audio = VAD(patient_audio, sr)
    # 3. 滑动窗口分割
    windows = create_sliding_windows(no_silence_audio, sr,window_size=window_size, stride=stride)

    # 4. 保存处理后的音频片段
    # os.makedirs(output_dir, exist_ok=True)
    #
    # for i, window in enumerate(windows):
    #     output_path = os.path.join(output_dir, f'segment_{i:04d}.wav')
    #     wav.write(output_path, sr, window.astype(np.float32))

    # 5. 提取特征
    features = extract_feature(windows)

    # 6. 保存特征

    np.save(os.path.join(output_dir, 'audio', 'origin', f'{ID}_features.npy'), features)

    print("特征提取完成")

    return len(windows)


# 使用示例
if __name__ == "__main__":
    # audio_path = "/home/zlh/DAIC/301_P/301_AUDIO.wav"
    # timestamps_path = "/home/zlh/DAIC/301_P/301_TRANSCRIPT.csv"
    # output_dir = "/home/zlh/DAIC/301_P"
    #
    # text_df = pd.read_csv(timestamps_path, sep='\t').fillna('')
    # window_size = 7
    # stride = 0
    #
    # n_segments = process_audio_file(audio_path, text_df, output_dir, window_size=window_size, stride=stride)
    # print(f"处理完成，共生成 {n_segments} 个音频片段")
    root  = '/home/zlh'
    root_dir = os.path.join(root, 'DAIC_WOZ-generated_database_4', 'train')
    create_folders(root_dir)
    # read training gt file
    gt_path = '/home/zlh/DAIC/train_split_Depression_AVEC2017.csv'
    gt_df = pd.read_csv(gt_path)
    window_size = 4
    stride = 1
    GT = {'clipped_data':
            {'no_gender_balance':
                {'ID_gt': [], 'gender_gt': [], 'phq_binary_gt': [], 'phq_score_gt': []}}}

    for i in range (len(gt_df)):

        # extract training gt details
        patient_ID = gt_df['Participant_ID'][i]
        phq_binary_gt = gt_df['PHQ8_Binary'][i]
        phq_score_gt = gt_df['PHQ8_Score'][i]
        gender_gt = gt_df['Gender'][i]
        phq_subscores_gt = gt_df.iloc[i, 4:].to_numpy().tolist()
        print(f'Processing Participant {patient_ID}, Gender: {gender_gt} ...')
        print(f'- PHQ Binary: {phq_binary_gt}, PHQ Score: {phq_score_gt}, Subscore: {phq_subscores_gt}')

        # get all files path of participant
        text_path = f'/home/zlh/DAIC/{patient_ID}_P/{patient_ID}_TRANSCRIPT.csv'
        audio_path = f'/home/zlh/DAIC/{patient_ID}_P/{patient_ID}_AUDIO.wav'

        # read text file
        text_df = pd.read_csv(text_path, sep='\t').fillna('')

        ##################################################################
        # start creating data in 'clipped_data/no_gender_balance' folder #
        ##################################################################

        output_root = os.path.join(root_dir, 'clipped_data', 'no_gender_balance')
        n_segments = process_audio_file(audio_path, text_df, output_root,ID= patient_ID ,window_size=window_size, stride=stride)
        print(f"处理完成，共生成 {n_segments} 个音频片段")

        # replicate GT
        # replicate GT
        for _ in range(n_segments):
            GT['clipped_data']['no_gender_balance']['ID_gt'].append(patient_ID)
            GT['clipped_data']['no_gender_balance']['gender_gt'].append(gender_gt)
            GT['clipped_data']['no_gender_balance']['phq_binary_gt'].append(phq_binary_gt)
            GT['clipped_data']['no_gender_balance']['phq_score_gt'].append(phq_score_gt)

    # store new GT
    for k1, v1 in GT.items():
        for k2, v2 in v1.items():
            for k3, v3 in v2.items():
                # print(os.path.join(root_dir, k1, k2, f'{k3}.npy'))
                np.save(os.path.join(root_dir, k1, k2, f'{k3}.npy'), v3)

