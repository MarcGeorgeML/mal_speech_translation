import os
import soundfile as sf
import numpy as np

def merge_wav_files(input_dir="temp_chunks", output_path="merged.wav"):
    # List all .wav files and sort them by chunk index
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    wav_files.sort(key=lambda x: int(x.split('_')[1]))  # Assumes 'chunk_{i}_processed.wav'

    merged_audio = []
    sr = None

    for wav_file in wav_files:
        file_path = os.path.join(input_dir, wav_file)
        audio, file_sr = sf.read(file_path)
        if sr is None:
            sr = file_sr
        elif sr != file_sr:
            raise ValueError(f"Sample rate mismatch: {file_path} has {file_sr}, expected {sr}")
        merged_audio.append(audio)

    merged_audio = np.concatenate(merged_audio)
    sf.write(output_path, merged_audio, sr)
    print(f"Merged {len(wav_files)} files into {output_path}")

