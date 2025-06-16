import librosa
import numpy as np
import soundfile as sf
import os
import importlib
import preprocess 
importlib.reload(preprocess)
from preprocess import preprocess_audio

def process_and_save(args):
    i, chunk, sr, temp_dir = args
    processed_audio, _ = preprocess_audio(chunk, sr, None)
    out_path = os.path.join(temp_dir, f"chunk_{i}_processed.wav")
    sf.write(out_path, processed_audio, sr)
    return out_path

def split_audio(audio, sr, chunk_duration=30):
    total_samples = len(audio)
    chunk_samples = int(chunk_duration * sr)
    if total_samples <= chunk_samples:
        return [audio], sr
    chunks = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        if len(chunk) < chunk_samples:
            pad_width = chunk_samples - len(chunk)
            chunk = np.pad(chunk, (0, pad_width), mode='constant')
        chunks.append(chunk)
    return chunks, sr

def process_chunks(input_path, temp_dir="temp_chunks", chunk_duration=30):
    import concurrent.futures

    os.makedirs(temp_dir, exist_ok=True)
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    chunks, sr = split_audio(audio, sr, chunk_duration)
    args_list = [(i, chunk, sr, temp_dir) for i, chunk in enumerate(chunks)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        chunk_paths = list(executor.map(process_and_save, args_list))

    print(f"Processed and saved {len(chunk_paths)} chunks to {temp_dir}")
    return chunk_paths, sr