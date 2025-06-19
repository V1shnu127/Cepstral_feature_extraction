#!/bin/env/python3
!pip install spafe

import os
import numpy as np
import librosa
from spafe.features.gfcc import gfcc
from scipy.io import savemat
from tqdm import tqdm

# ----------- Delta Calculation Function -----------
# def compute_deltas(x, hlen=1):
#     win = np.arange(hlen, -hlen - 1, -1)
#     padded = np.pad(x, ((0, 0), (hlen, hlen)), mode='edge')
#     delta = np.apply_along_axis(lambda m: np.convolve(m, win, mode='valid'), axis=1, arr=padded)
#     delta = delta / (2 * np.sum(np.arange(1, hlen + 1) ** 2))
#     return delta

# ------------------ Settings ------------------
base_dataset_dir = "/kaggle/input/segmented-bc2-ttv-19-june-2025"
save_base_dir = "/kaggle/working/gfcc_bc2_ttv_2"

dataset_splits = {
    "train": "train",
    "val": "val",
    "test": "test"
}

feature_type = "gfcc"

# ------------------ Feature Extraction ------------------
feature_extractors = {
    "mfcc": mfcc,
    "lfcc": lfcc,
    "gfcc": gfcc
}

for split_name, folder_name in dataset_splits.items():
    base_folder = os.path.join(base_dataset_dir, folder_name)
    save_dir = os.path.join(save_base_dir, feature_type, split_name)

    wav_files = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    print(f"Processing {split_name.upper()} - Found {len(wav_files)} audio files.")

    for file_path in tqdm(wav_files, desc=f"Extracting {feature_type.upper()} for {split_name.upper()}"):
        try:
            sig, fs = librosa.load(file_path, sr=16000)
            
            # Skip files shorter than 0.01 seconds
            if len(sig) / fs < 0.01:
                print(f"Skipping {file_path}: Duration {len(sig)/fs} seconds is too short.")
                with open("errors.log", "a") as log_file:
                    log_file.write(f"Skipping {file_path}: Duration {len(sig)/fs} seconds is too short.\n")
                continue

            # Extract base features (remove win_len and win_hop for gfcc)
            base_feats = feature_extractors[feature_type](sig, fs=fs, num_ceps=20).T

            # Save path handling
            relative_path = os.path.relpath(file_path, base_folder)
            save_path = os.path.join(save_dir, os.path.dirname(relative_path))
            os.makedirs(save_path, exist_ok=True)

            file_name = os.path.splitext(os.path.basename(file_path))[0]
            mat_file = os.path.join(save_path, f"{file_name}.mat")

            # Save as .mat
            savemat(mat_file, {'final': base_feats})

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            with open("errors.log", "a") as log_file:
                log_file.write(f"Error processing {file_path}: {e}\n")
            continue

print(f"{feature_type.upper()} Feature extraction complete!")
