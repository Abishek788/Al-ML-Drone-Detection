# import glob
# import numpy as np
# import pandas as pd  # This import is unused but harmless
# import soundfile
# from python_speech_features import fbank
# from scipy.signal import spectrogram  # This import is unused but harmless
# from os import path

# def rescale_for_model(fbanks):
#     pass


# def make_fbank(wav, fs=22050):
#     """
#     Converts raw audio (wav) into a Mel-frequency filter bank (fbank).
#     This is the audio 'fingerprint' the model will learn from.
#     """
#     winlen = 1. / 43.0664 # specRes_Hz from model 
#     winstep = 2.9 / 1000. # tempRes_ms from model
#     nfft = 1024
#     preemph = 0.5
#     M, _ = fbank(wav, samplerate=fs,
#                  nfilt=41, nfft=nfft,
#                  lowfreq=0, highfreq=11025,
#                  preemph=0.5,
#                  winlen=winlen, winstep=winstep,
#                  winfunc=lambda x: np.hanning(x))

#     logM = np.log(M)
#     logM = np.swapaxes(logM, 0, 1)

#     # --- THIS IS THE CRITICAL CHANGE ---
#     # The original was 682 for 2-second clips.
#     # For 0.1-second clips, we use a much smaller target size.
#     targetSize = 35 
#     # --- END OF CHANGE ---

#     cut = np.minimum(logM.shape[1], targetSize)
#     background = np.float64(logM[:,:cut]).mean(axis=1)

#     features = np.float32(np.float64(logM) - background[:, np.newaxis])

#     # Pad or truncate to the new targetSize
#     if features.shape[1] < targetSize:
#         features = np.concatenate((features,
#                                    np.zeros((features.shape[0],
#                                              targetSize-features.shape[1]),
#                                             dtype='float32')), axis=1)
#     elif features.shape[1] > targetSize:
#         features = features[:,:(targetSize-features.shape[1])]

#     return features

# def load_drone_data(drone_dir, nondrone_dir):
#     """
#     Loads pre-clipped audio from the DroneAudioDataset.
#     'Drone' files are labeled 1, 'Non-Drone' are labeled 0.
#     """
#     data = []
#     target = []
    
#     print("Processing drone files...")
#     # Use path.join for robust file paths
#     drone_files = glob.glob(path.join(drone_dir, "*.wav"))
#     if not drone_files:
#         print(f"Warning: No .wav files found in {drone_dir}")
        
#     for f in drone_files:
#         wav, fs = soundfile.read(f)
#         if np.ndim(wav) > 1:
#             wav = wav[:,0] # Handle stereo, take one channel
        
#         # We store the audio, the path (for debugging), and a 0 window-ID
#         data.append([wav, f, 0])
#         target.append(1)

#     print(f"Found {len(drone_files)} drone files.")

#     print("Processing non-drone files...")
#     nondrone_files = glob.glob(path.join(nondrone_dir, "*.wav"))
#     if not nondrone_files:
#         print(f"Warning: No .wav files found in {nondrone_dir}")

#     for f in nondrone_files:
#         wav, fs = soundfile.read(f)
#         if np.ndim(wav) > 1:
#             wav = wav[:,0]
            
#         data.append([wav, f, 0])
#         target.append(0)

#     print(f"Found {len(nondrone_files)} non-drone files.")
    
#     # Return in the same format as the original preprocess.py
#     # dtype=object is needed because the array holds mixed types (numpy arrays and strings)
#     return np.array(data, dtype=object), np.array(target, dtype=np.int8)

import glob
import numpy as np
import pandas as pd
import soundfile
from python_speech_features import fbank
from scipy.signal import spectrogram
from os import path

def rescale_for_model(fbanks):
    pass


def make_fbank(wav, fs=22050):
    """
    Converts raw audio (wav) into a Mel-frequency filter bank (fbank).
    This is the audio 'fingerprint' the model will learn from.
    """
    winlen = 1. / 43.0664 # specRes_Hz from model 
    winstep = 2.9 / 1000. # tempRes_ms from model
    nfft = 1024
    preemph = 0.5
    M, _ = fbank(wav, samplerate=fs,
                 nfilt=41, nfft=nfft,
                 lowfreq=0, highfreq=11025,
                 preemph=0.5,
                 winlen=winlen, winstep=winstep,
                 winfunc=lambda x: np.hanning(x))

    logM = np.log(M)
    logM = np.swapaxes(logM, 0, 1)

    # --- THIS IS THE CRITICAL CHANGE ---
    # The original was 682 for 2-second clips.
    # For 0.1-second clips, we use a much smaller target size.
    targetSize = 35 
    # --- END OF CHANGE ---

    cut = np.minimum(logM.shape[1], targetSize)
    background = np.float64(logM[:,:cut]).mean(axis=1)

    features = np.float32(np.float64(logM) - background[:, np.newaxis])

    # Pad or truncate to the new targetSize
    if features.shape[1] < targetSize:
        features = np.concatenate((features,
                                   np.zeros((features.shape[0],
                                             targetSize-features.shape[1]),
                                            dtype='float32')), axis=1)
    elif features.shape[1] > targetSize:
        features = features[:,:(targetSize-features.shape[1])]

    return features

# --- NEW FUNCTION ---
# This replaces the old load_drone_data function
def load_multiclass_data(base_dir):
    """
    Loads pre-clipped audio from a multi-class directory structure.
    Expects subfolders like '0_ClassName', '1_ClassName', etc.
    The label is taken from the number at the start of the folder name.
    """
    data = []
    target = []
    
    # Find all subdirectories
    class_folders = glob.glob(path.join(base_dir, "*/"))
    
    if not class_folders:
        print(f"Error: No class folders found in {base_dir}")
        return np.array(data, dtype=object), np.array(target, dtype=np.int8)

    print(f"Found {len(class_folders)} class folders...")

    for folder_path in class_folders:
        folder_name = path.basename(path.normpath(folder_path))
        
        # Get the label from the folder name
        try:
            label = int(folder_name.split('_')[0])
            print(f"Processing folder: {folder_name} (Label: {label})")
        except ValueError:
            print(f"Skipping folder (invalid name): {folder_name}")
            continue

        # Load all .wav files from this folder
        wav_files = glob.glob(path.join(folder_path, "*.wav"))
        print(f"Found {len(wav_files)} files.")
        
        for f in wav_files:
            try:
                wav, fs = soundfile.read(f)
                if np.ndim(wav) > 1:
                    wav = wav[:,0]
                
                data.append([wav, f, 0]) # [audio, filepath, window_id]
                target.append(label)
            except Exception as e:
                print(f"Warning: Could not read {f}. Error: {e}")

    # Return in the same format
    return np.array(data, dtype=object), np.array(target, dtype=np.int8)