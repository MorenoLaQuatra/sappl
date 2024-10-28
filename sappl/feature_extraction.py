# sappl/feature_extraction.py

import numpy as np
import librosa

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def extract_mfcc(audio, sample_rate=16000, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from an audio signal.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): Input audio array.
        sample_rate (int): Sample rate of the audio. Default is 16000.
        n_mfcc (int): Number of MFCCs to extract. Default is 13.
        n_fft (int): Length of the FFT window. Default is 2048.
        hop_length (int): Number of samples between successive frames. Default is 512.

    Returns:
        np.ndarray: MFCCs of shape (T, n_mfcc), where T is the number of time steps.
    """
    if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc.T  # Transpose to (T, n_mfcc)


def extract_chroma(audio, sample_rate=16000, n_fft=2048, hop_length=512):
    """
    Extracts chroma features from an audio signal, which represent the energy distribution across pitch classes.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): Input audio array.
        sample_rate (int): Sample rate of the audio. Default is 16000.
        n_fft (int): Length of the FFT window. Default is 2048.
        hop_length (int): Number of samples between successive frames. Default is 512.

    Returns:
        np.ndarray: Chroma features of shape (T, 12), where T is the number of time steps.
    """
    if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    return chroma.T  # Transpose to (T, 12)


def extract_tonnetz(audio, sample_rate=16000):
    """
    Extracts tonnetz (tonal centroid features) from an audio signal, which capture harmonic relations.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): Input audio array.
        sample_rate (int): Sample rate of the audio. Default is 16000.

    Returns:
        np.ndarray: Tonnetz features of shape (T, 6), where T is the number of time steps.
    """
    if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
    return tonnetz.T  # Transpose to (T, 6)


def extract_zero_crossing_rate(audio, frame_length=2048, hop_length=512):
    """
    Computes the zero-crossing rate of the audio signal.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): Input audio array.
        frame_length (int): Length of the frame over which to compute zero-crossing rate. Default is 2048.
        hop_length (int): Number of samples between successive frames. Default is 512.

    Returns:
        np.ndarray: Zero-crossing rate of shape (T,), where T is the number of time steps.
    """
    if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)
    return zcr.T  # Transpose to (T, 1)


def extract_spectral_contrast(audio, sample_rate=16000, n_fft=2048, hop_length=512, n_bands=6):
    """
    Computes the spectral contrast of the audio signal, representing the difference between peaks and valleys.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): Input audio array.
        sample_rate (int): Sample rate of the audio. Default is 16000.
        n_fft (int): Length of the FFT window. Default is 2048.
        hop_length (int): Number of samples between successive frames. Default is 512.
        n_bands (int): Number of frequency bands. Default is 6.

    Returns:
        np.ndarray: Spectral contrast features of shape (T, n_bands + 1), where T is the number of time steps.
    """
    if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands)
    return spectral_contrast.T  # Transpose to (T, n_bands + 1)


if __name__ == "__main__":
    # Example usage for testing purposes
    from sappl.io import load_audio

    # Load an example audio file
    test_audio_path = "../samples/music_sample.wav"  # Replace with a valid path
    audio_data = load_audio(test_audio_path, sample_rate=16000, mono=True)

    # Extract MFCC features
    mfcc = extract_mfcc(audio_data, sample_rate=16000)
    print("MFCC shape:", mfcc.shape)

    # Extract Chroma features
    chroma = extract_chroma(audio_data, sample_rate=16000)
    print("Chroma shape:", chroma.shape)

    # Extract Tonnetz features
    tonnetz = extract_tonnetz(audio_data, sample_rate=16000)
    print("Tonnetz shape:", tonnetz.shape)

    # Extract Zero-Crossing Rate
    zcr = extract_zero_crossing_rate(audio_data)
    print("Zero-Crossing Rate shape:", zcr.shape)

    # Extract Spectral Contrast
    spectral_contrast = extract_spectral_contrast(audio_data, sample_rate=16000)
    print("Spectral Contrast shape:", spectral_contrast.shape)
