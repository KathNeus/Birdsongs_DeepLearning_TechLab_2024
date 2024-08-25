#!pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 -f https://storage.googleapis.com/libtpu-releases/index.html

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

torch.cuda.empty_cache()

# Split the augmented dataset
train_size = int(0.6 * len(augmented_dataset))
valid_size = int(0.2 * len(augmented_dataset))
test_size = len(augmented_dataset) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = random_split(augmented_dataset, [train_size, valid_size, test_size])

# DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=2, pin_memory=True)

# Print sample sizes
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Function to compute auto-correlation
def compute_auto_correlation(spectrogram):
    autocorr = np.correlate(spectrogram, spectrogram, mode='full')
    return autocorr[autocorr.size // 2:]

# Print auto-correlation statistics
def print_auto_correlation_stats(loader, num_samples=5):
    for i, (spectrograms, labels) in enumerate(loader):
        if i >= num_samples:
            break
        for spectrogram in spectrograms:
            spectrogram_np = spectrogram.numpy().flatten()
            autocorr = compute_auto_correlation(spectrogram_np)
            print(f"Auto-correlation (sample {i}): Mean={np.mean(autocorr)}, Std={np.std(autocorr)}")

# Compute and print auto-correlation statistics for a few samples from each set
print("Training set auto-correlation statistics:")
#print_auto_correlation_stats(train_loader)

print("Validation set auto-correlation statistics:")
#print_auto_correlation_stats(valid_loader)

print("Test set auto-correlation statistics:")
#print_auto_correlation_stats(test_loader)

#############################################
### Preview random batches
#############################################

import matplotlib.pyplot as plt
import librosa.display

def pad_spectrogram(spectrogram, sr=44100, n_fft=1024, min_freq=1024, max_freq=8192):
    # Calculate the FFT frequencies
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Determine the indices for the minimum and maximum frequencies
    min_idx = np.argmax(freqs >= min_freq)
    max_idx = np.argmax(freqs > max_freq)

    # Calculate the range of the source frequency bins to be copied
    source_freq_range = max_idx - min_idx

    # Truncate the frequencies to the required range
    freqs = freqs[min_idx:max_idx]

    # Create a new padded spectrogram with 513 frequency bins
    padded_spectrogram = np.zeros((513, spectrogram.shape[1]))

    # Calculate the range of the target frequency bins
    target_min_idx = (min_freq / (sr / 2)) * (n_fft // 2)
    target_max_idx = target_min_idx + source_freq_range

    # Map the original spectrogram to the padded spectrogram
    pad_start_idx = int(np.floor(target_min_idx))
    pad_end_idx = int(np.ceil(target_max_idx))

    # Fill the padded spectrogram with the original data
    #print("padded_spectrogram[pad_start_idx:pad_end_idx, :].shape:"+str(padded_spectrogram[pad_start_idx:pad_end_idx, :].shape))
    #print("spectrogram.shape:"+str(spectrogram.shape))
    padded_spectrogram[pad_start_idx:pad_end_idx-2, :] = spectrogram

    return padded_spectrogram

def cut_off_freqs(spectrogram,min_freq,max_freq,frequencies):
  # Find the frequency bin indices for the desired range
     cutoffspec=spectrogram
     min_bin = np.argmax(frequencies >= min_freq)
     max_bin = np.argmax(frequencies > max_freq)
     cutoffspec[0][:min_bin, :]=0
     cutoffspec[0][max_bin:, :]=0
     return cutoffspec

# Function to visualize a batch of spectrograms
def visualize_spectrogram_batch(spectrograms, labels, title="Spectrograms", sr=44100, hop_length=512, fmin=1024, fmax=8192):
    plt.figure(figsize=(15, 16))
    for i in range(len(spectrograms)):
        plt.subplot(3, 3, i + 1)
        spectrogram = spectrograms[i].numpy()
        #spectrogram=
        label = label_encoder.inverse_transform([labels[i]])[0]
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=1024)
        #spectrogram = cut_off_freqs(spectrogram,fmin,fmax,frequencies)
        librosa.display.specshow(librosa.power_to_db(pad_spectrogram(spectrogram[0]), ref=np.max), sr=sr, hop_length=hop_length,  fmin=fmin, fmax=fmax, x_axis='time',y_axis='linear')

        plt.colorbar(format='%+2.0f dB')
        plt.title(f'{title} - {label}')
        plt.tight_layout()
        plt.ylim(fmin*0.8,fmax*1.2)
    #print(plt.ylim())
    plt.show()

# Function to get a random batch of samples from a DataLoader
def get_random_batch(loader, num_samples=3):
    data_iter = iter(loader)
    spectrograms, labels = next(data_iter)
    return spectrograms[:num_samples], labels[:num_samples]

# Get random samples from train, validation, and test loaders
train_spectrograms, train_labels = get_random_batch(train_loader)
valid_spectrograms, valid_labels = get_random_batch(valid_loader)
test_spectrograms, test_labels = get_random_batch(test_loader)

# Visualize samples from each dataset
print("Training set samples:")
visualize_spectrogram_batch(train_spectrograms, train_labels, title="Train")

print("Validation set samples:")
visualize_spectrogram_batch(valid_spectrograms, valid_labels, title="Valid")

print("Test set samples:")
visualize_spectrogram_batch(test_spectrograms, test_labels, title="Test")