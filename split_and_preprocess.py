import librosa
import numpy as np
import soundfile as sf
import os
import importlib
import preprocess 
importlib.reload(preprocess)
from preprocess import preprocess_audio

def split_audio(input_path, chunk_duration=30):
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    total_samples = len(audio)
    chunk_samples = int(chunk_duration * sr)
    # If audio is shorter than chunk_duration, return as a single chunk
    if total_samples <= chunk_samples:
        return [audio], sr
    chunks = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        if len(chunk) < chunk_samples:
            # Optionally pad the last chunk
            pad_width = chunk_samples - len(chunk)
            chunk = np.pad(chunk, (0, pad_width), mode='constant')
        chunks.append(chunk)
    return chunks, sr

def process_chunks(input_path, output_path, temp_dir="temp_chunks", chunk_duration=30):
    os.makedirs(temp_dir, exist_ok=True)
    chunks, sr = split_audio(input_path, chunk_duration)
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        temp_in = os.path.join(temp_dir, f"chunk_{i}.wav")
        temp_out = os.path.join(temp_dir, f"chunk_{i}_processed.wav")
        sf.write(temp_in, chunk, sr)
        processed_audio, _ = preprocess_audio(temp_in, temp_out)
        processed_chunks.append(processed_audio)
    # Concatenate all processed chunks
    merged_audio = np.concatenate(processed_chunks)
    sf.write(output_path, merged_audio, sr)
    print(f"Saved merged processed audio to {output_path}")
    return merged_audio, sr
    
# split_audio("test2.wav", chunk_duration=30)
# process_chunks("test2.wav", "output.wav", chunk_duration=30)