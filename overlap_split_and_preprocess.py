import os
import numpy as np
import soundfile as sf
from preprocess import preprocess_audio

def split_preprocess_and_save_chunks(audio, sr, chunk_duration=30, overlap_duration=5, temp_dir="temp_dir"):
    """
    Splits audio into overlapping chunks, preprocesses each chunk, saves as .wav in temp_dir,
    and returns a list of all .wav file paths.
    """
    os.makedirs(temp_dir, exist_ok=True)
    total_samples = len(audio)
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap_duration * sr)
    wav_paths = []
    start = 0
    idx = 0
    while start < total_samples:
        end = start + chunk_samples
        chunk = audio[start:end]
        if len(chunk) < chunk_samples:
            pad_width = chunk_samples - len(chunk)
            chunk = np.pad(chunk, (0, pad_width), mode='constant')
        chunk = chunk.astype(np.float32)
        # Preprocess the chunk
        processed_chunk, _ = preprocess_audio(chunk, sr)
        # Save as .wav file
        out_path = os.path.join(temp_dir, f"chunk_{idx}.wav")
        sf.write(out_path, processed_chunk, sr)
        wav_paths.append(out_path)
        start += chunk_samples - overlap_samples
        idx += 1
    return wav_paths