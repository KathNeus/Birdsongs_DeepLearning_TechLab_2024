import torch
import torchaudio
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

class AugmentedBirdsongDataset(BirdsongDataset):
    def __init__(self, spectrogram_dir, length, label_encoder, augmentation_settings):
        super().__init__(spectrogram_dir, length, label_encoder)
        self.augment_prob = augmentation_settings.get('augment_prob', 0) #0.5
        self.overlay_prob = augmentation_settings.get('overlay_prob', 0) #0.33
        self.noise_prob = augmentation_settings.get('noise_prob', 0) #0.33
        self.shift_prob = augmentation_settings.get('shift_prob', 0) #0.33
        self.noise_level = augmentation_settings.get('noise_level', 0) #0.1
        self.max_shift = augmentation_settings.get('max_shift', 0) #0.25

    def __getitem__(self, idx):
        spectrogram, label = super().__getitem__(idx)

        # Apply augmentations based on augment_prob
        if random.random() < self.augment_prob:
            spectrogram = self.apply_augmentations(spectrogram)

        # Normalize the spectrogram after augmentation
        spectrogram = self.normalize_spectrogram(spectrogram)

        return spectrogram, label

    def apply_augmentations(self, spectrogram):
        # Overlay another sample with a given probability
        if random.random() < self.overlay_prob:
            spectrogram = self.overlay_samples(spectrogram)
        # Add noise to the sample with a given probability
        if random.random() < self.noise_prob:
            spectrogram = self.add_noise(spectrogram)
        # Shift the sample in time with a given probability
        if random.random() < self.shift_prob:
            spectrogram = self.shift_time(spectrogram)
        return spectrogram

    def overlay_samples(self, spectrogram):
        # Select another random sample to overlay
        idx = random.randint(0, len(self.files) - 1)
        overlay_spectrogram, _ = super().__getitem__(idx)

        alpha = random.uniform(0.1, 0.5)  # Amplitude of the overlayed sample
        beta = 1 - alpha  # Amplitude of the original sample
        return alpha * spectrogram + beta * overlay_spectrogram

    def add_noise(self, spectrogram):
        noise = torch.randn(spectrogram.size()) * self.noise_level  # Add Gaussian noise
        return spectrogram + noise

    def shift_time(self, spectrogram):
        shift_amount = random.randint(0, int(self.max_shift * spectrogram.size(2)))
        direction = random.choice(['left', 'right'])

        if direction == 'left':
            # Shift spectrogram left and pad with zeros on the right
            pad = torch.zeros_like(spectrogram[:, :, :shift_amount])
            spectrogram = torch.cat((spectrogram[:, :, shift_amount:], pad), dim=2)
        else:
            # Shift spectrogram right and pad with zeros on the left
            pad = torch.zeros_like(spectrogram[:, :, :shift_amount])
            spectrogram = torch.cat((pad, spectrogram[:, :, :-shift_amount]), dim=2)

        return spectrogram

    def normalize_spectrogram(self, spectrogram):
        # Replace NaNs and Infs with zeros
        spectrogram = torch.where(torch.isnan(spectrogram), torch.zeros_like(spectrogram), spectrogram)
        spectrogram = torch.where(torch.isinf(spectrogram), torch.zeros_like(spectrogram), spectrogram)
        #mean = torch.mean(spectrogram)
        #std = torch.std(spectrogram)
        return spectrogram#return (spectrogram - mean) / std

# Define augmentation settings in a dictionary
augmentation_settings = {
    'augment_prob': 0,  # Probability of applying any augmentation 0.5
    'overlay_prob': 0,  # Probability of overlaying another sample 0.0
    'noise_prob': 0,  # Probability of adding noise 0.15
    'shift_prob': 0,  # Probability of shifting the sample in time 0.15
    'noise_level': 0,  # Magnitude of the noise added 0.00001
    'max_shift': 0 # Maximum fraction of the spectrogram length to shift 0.15
}

# Use the augmented dataset for training
print("length=", max_length)
augmented_dataset = AugmentedBirdsongDataset(
    output_directory,
    length=max_length,
    label_encoder=label_encoder,
    augmentation_settings=augmentation_settings
)