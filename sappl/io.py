import numpy as np
import soundfile as sf
import librosa

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_audio(file_path, sample_rate=16000, mono=True):
    """
    Loads an audio file from the specified path, resamples to the target sample rate if necessary, and converts to mono.

    Args:
        file_path (str): Path to the audio file.
        sample_rate (int): Target sample rate for resampling. Default is 16000.
        mono (bool): If True, converts the audio to mono. Default is True.

    Returns:
        np.ndarray: Loaded audio data as a NumPy array of shape (samples,).
    
    Raises:
        FileNotFoundError: If the audio file is not found at the specified path.
        ValueError: If the file cannot be loaded or is in an unsupported format.
        ImportError: If PyTorch tensors are passed but PyTorch is not installed.
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string representing the path to the audio file.")
    
    try:
        # Load audio with the original sampling rate
        audio, orig_sr = librosa.load(file_path, sr=None, mono=mono)
    except FileNotFoundError:
        raise FileNotFoundError(f"Audio file not found at {file_path}. Please provide a valid file path.")
    except Exception as e:
        raise ValueError(f"Failed to load audio file {file_path}. Error: {e}")
    
    # Resample to the target sample rate if necessary
    if orig_sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sample_rate)

    return audio


def save_audio(file_path, audio, sample_rate=16000):
    """
    Saves an audio array to a file, supporting multiple audio formats. Converts PyTorch tensors to NumPy arrays if needed.

    Args:
        file_path (str): Path where the audio file should be saved (include extension like .wav, .mp3).
        audio (Union[np.ndarray, torch.Tensor]): Audio data to save.
        sample_rate (int): Sample rate of the audio. Default is 16000.
    
    Raises:
        ValueError: If the audio data is not in a valid format or file extension is unsupported.
        ImportError: If audio is a PyTorch tensor but PyTorch is not installed.
    """
    # Check if audio is a PyTorch tensor and convert to NumPy
    if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    elif isinstance(audio, np.ndarray):
        audio = audio
    else:
        raise ValueError("Audio data must be either a NumPy array or a PyTorch tensor.")

    if audio.ndim > 1:
        raise ValueError("Audio data should be a 1D array for mono or 2D for stereo format.")

    # Check for supported file extensions
    if not file_path.lower().endswith(('.wav', '.flac', '.mp3', '.ogg')):
        raise ValueError("Unsupported file format. Please use a supported format: .wav, .flac, .mp3, .ogg.")

    try:
        sf.write(file_path, audio, sample_rate)
    except Exception as e:
        raise ValueError(f"Failed to save audio to {file_path}. Error: {e}")


if __name__ == "__main__":
    test_load_path = "../samples/music_sample.wav"  # Replace with an actual path for loading
    test_save_path = "../samples/music_sample_saved.wav"  # Replace with an actual path for saving

    # Test loading audio
    try:
        audio_data = load_audio(test_load_path, sample_rate=16000, mono=True)
        print("Loaded audio shape:", audio_data.shape)
    except Exception as e:
        print("Error loading audio:", e)

    # Test saving audio
    try:
        save_audio(test_save_path, audio_data, sample_rate=16000)
        print(f"Audio successfully saved to {test_save_path}")
    except Exception as e:
        print("Error saving audio:", e)