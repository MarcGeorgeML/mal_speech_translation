import librosa
from scipy.signal import butter, filtfilt
import numpy as np
import os
import matplotlib.pyplot as plt 
import sounddevice as sd
import noisereduce as nr
import soundfile as sf

def preprocess_audio(
    input,
    s,
    output_path=None
):
    
    # trimming the audio to remove silence
    trim_audio, _= librosa.effects.trim(input)

    # slowing down or speeding up the audio
    def estimate_speech_rate(y, sr):
        # Detect onsets (approximates syllables)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        
        # Estimate speech rate as onsets per second
        duration = librosa.get_duration(y=y, sr=sr)
        rate = len(onsets) / duration
        return rate

    def process_speed(y, sr):
        original_len = len(y)
        rate = estimate_speech_rate(y, sr)
        print(f"Estimated speech rate: {rate:.2f} onsets/sec")

        if rate > 8.0:
            print("Speech too fast. Slowing down...")
            y = librosa.effects.time_stretch(y, rate=0.5)
        elif rate < 2.5:
            print("Speech too slow. Speeding up...")
            y = librosa.effects.time_stretch(y, rate=1.5)
        else:
            print("Speech rate is normal.")

        print(f"Original length: {original_len}, Processed length: {len(y)}")
        return y

    tempo_audio = process_speed(trim_audio, s)


    # Design a band-pass filter for the midrange (1khz - 8khz)
    nyquist = 0.5 * s
    low = 1000 / nyquist
    high = 8000 / nyquist
    b, a = butter(4, [low, high], btype='band')
    midrange = filtfilt(b, a, tempo_audio)
    # Boost midrange and mix with original
    boosted = tempo_audio + 0.2 * midrange

    # apply high-pass filter to remove low-frequency noise(60hz)
    nyquist = 0.5 * s
    normal_cutoff = 20 / nyquist
    b, a = butter(4, normal_cutoff, btype='high', analog=False)
    hp_filtered_audio = filtfilt(b, a, tempo_audio)

    # apply low-pass filter to remove high-frequency noise(10khz)
    nyquist = 0.5 * s
    normal_cutoff = 10000 / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    lp_filtered_audio = filtfilt(b, a, hp_filtered_audio)        

    # Boost volume by a factor (e.g., 2.0 for double the volume)
    gain = 1.7
    boosted_audio = lp_filtered_audio * gain

    # applying clip to prevent clipping
    clip_audio = np.clip(boosted_audio, -1.0, 1.0)
    norm_audio = librosa.util.normalize(clip_audio)


    # Calculate RMS
    rms = np.sqrt(np.mean(norm_audio**2))
    # Convert RMS to dBFS (decibels relative to full scale)
    db = 20 * np.log10(rms)
    # Print RMS and dB level
    print(f"RMS dB level: {db:.2f} dBFS")

    return norm_audio, s
