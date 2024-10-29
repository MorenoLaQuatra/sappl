import numpy as np
import librosa

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def stft(audio, n_fft=2048, hop_length=512, win_length=None):
    """
    Computes the Short-Time Fourier Transform (STFT) of the audio signal.

    Args:
        audio (np.ndarray): Input audio array.
        n_fft (int): Number of FFT components. Default is 2048.
        hop_length (int): Number of audio samples between successive STFT columns. Default is 512.
        win_length (int): Each frame of audio will be windowed by `win_length`. Default is `n_fft`.

    Returns:
        np.ndarray: STFT of shape (T, F), where T is the number of time steps, and F is the number of frequency bins.
    """
    stft_matrix = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return stft_matrix.T  # Transpose to (T, F)


def istft(stft_matrix, hop_length=512, win_length=None):
    """
    Computes the inverse Short-Time Fourier Transform (iSTFT) to reconstruct the time-domain audio signal.

    Args:
        stft_matrix (np.ndarray): STFT matrix with shape (T, F).
        hop_length (int): Number of audio samples between successive STFT columns. Default is 512.
        win_length (int): Each frame of audio will be windowed by `win_length`. Default is None.

    Returns:
        np.ndarray: Reconstructed audio time series.
    """
    return librosa.istft(stft_matrix.T, hop_length=hop_length, win_length=win_length)  # Transpose back to (F, T) for iSTFT


def magphase(stft_matrix):
    """
    Separates the magnitude and phase of the STFT matrix.

    Args:
        stft_matrix (np.ndarray): STFT matrix of shape (T, F).

    Returns:
        tuple: (magnitude, phase) both of shape (T, F).
    """
    magnitude, phase = librosa.magphase(stft_matrix.T)
    return magnitude.T, phase.T  # Transpose each to (T, F)


def db_to_power(spectrogram_db, ref=1.0):
    """
    Converts a spectrogram from dB scale to power scale.

    Args:
        spectrogram_db (np.ndarray): Input spectrogram in dB scale (T, F).
        ref (float): Reference power value. Default is 1.0.

    Returns:
        np.ndarray: Spectrogram in power scale (T, F).
    """
    return librosa.db_to_power(spectrogram_db, ref=ref)


def power_to_db(spectrogram, ref=1.0):
    """
    Converts a spectrogram from power scale to dB scale.

    Args:
        spectrogram (np.ndarray): Input spectrogram in power scale (T, F).
        ref (float): Reference power value. Default is 1.0.

    Returns:
        np.ndarray: Spectrogram in dB scale (T, F).
    """
    return librosa.power_to_db(spectrogram.T, ref=ref).T  # Convert and transpose to (T, F)


def amplitude_to_db(spectrogram_amplitude, ref=1.0):
    """
    Converts an amplitude spectrogram to dB scale.

    Args:
        spectrogram_amplitude (np.ndarray): Input amplitude spectrogram (T, F).
        ref (float): Reference amplitude value. Default is 1.0.

    Returns:
        np.ndarray: Spectrogram in dB scale (T, F).
    """
    return librosa.amplitude_to_db(spectrogram_amplitude.T, ref=ref).T  # Convert and transpose to (T, F)


def db_to_amplitude(spectrogram_db, ref=1.0):
    """
    Converts a dB spectrogram to amplitude scale.

    Args:
        spectrogram_db (np.ndarray): Input spectrogram in dB scale (T, F).
        ref (float): Reference amplitude value. Default is 1.0.

    Returns:
        np.ndarray: Spectrogram in amplitude scale (T, F).
    """
    return librosa.db_to_amplitude(spectrogram_db, ref=ref)


def compute_mel_spectrogram(audio, sample_rate=16000, n_fft=2048, hop_length=512, n_mels=128, f_min=0.0, f_max=None):
    """
    Computes the Mel spectrogram of the audio signal.

    Args:
        audio (np.ndarray): Input audio array.
        sample_rate (int): Sampling rate of the audio. Default is 16000.
        n_fft (int): Number of FFT components. Default is 2048.
        hop_length (int): Number of audio samples between successive Mel spectrogram columns. Default is 512.
        n_mels (int): Number of Mel bands. Default is 128.
        f_min (float): Minimum frequency in Hz. Default is 0.0.
        f_max (float): Maximum frequency in Hz. Default is None (half the sampling rate).

    Returns:
        np.ndarray: Mel spectrogram in dB scale (T, F), where T is time steps and F is Mel bins.
    """
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max or sample_rate / 2
    )
    return librosa.power_to_db(mel_spectrogram, ref=np.max).T  # Convert to dB and transpose to (T, F)


def reconstruct_waveform(magnitude, phase, hop_length=512, win_length=None):
    """
    Reconstructs the waveform from magnitude and phase components.

    Args:
        magnitude (np.ndarray): Magnitude spectrogram (T, F).
        phase (np.ndarray): Phase spectrogram (T, F).
        hop_length (int): Number of audio samples between successive STFT columns. Default is 512.
        win_length (int): Each frame of audio will be windowed by `win_length`. Default is None.

    Returns:
        np.ndarray: Reconstructed audio waveform.
    """
    # Transpose magnitude and phase to (F, T) for compatibility with istft
    magnitude = magnitude.T
    phase = phase.T

    # Combine magnitude and phase to form the complex-valued STFT matrix
    stft_matrix = magnitude * np.exp(1j * phase)

    # Perform inverse STFT to get back the waveform
    return librosa.istft(stft_matrix, hop_length=hop_length, win_length=win_length)


if __name__ == "__main__":
    # Example usage for testing purposes
    from sappl.io import load_audio, save_audio

    # Load an example audio file
    test_audio_path = "../samples/music_sample.wav"  # Replace with a valid file path
    audio = load_audio(test_audio_path, sample_rate=16000, mono=True)
    print("Original audio shape:", audio.shape)
    print()
    
    # Compute and display STFT
    stft_matrix = stft(audio)
    print("STFT shape:", stft_matrix.shape)
    print()

    # Separate magnitude and phase
    magnitude, phase = magphase(stft_matrix)
    print("Magnitude shape:", magnitude.shape)
    print("Phase shape:", phase.shape)
    print()
    
    # Reconstruct waveform from magnitude and phase
    reconstructed_audio = reconstruct_waveform(magnitude, phase, hop_length=512)
    print("Reconstructed audio shape:", reconstructed_audio.shape)

    # Save the reconstructed audio
    test_save_path = "../samples/music_sample_reconstructed.wav"
    save_audio(test_save_path, reconstructed_audio, sample_rate=16000)
    print(f"Reconstructed audio saved to {test_save_path}")

    # Convert magnitude to dB scale
    magnitude_db = amplitude_to_db(magnitude)
    print("Magnitude in dB shape:", magnitude_db.shape)
    print()

    # Compute inverse STFT to reconstruct the audio
    reconstructed_audio = istft(stft_matrix)
    print("Reconstructed audio shape:", reconstructed_audio.shape)
    print()

    # Save the reconstructed audio
    test_save_path = "../samples/music_sample_reconstructed.wav"  # Replace with a valid save path
    save_audio(test_save_path, reconstructed_audio, sample_rate=16000)
    print(f"Reconstructed audio saved to {test_save_path}")
    print()
