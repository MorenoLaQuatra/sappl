import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def convert_to_mono(audio):
    """
    Converts a stereo audio array to mono by averaging the channels.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): Input audio array of shape (samples,) or (channels, samples).

    Returns:
        Union[np.ndarray, torch.Tensor]: Mono audio array of shape (samples,).
    """
    if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
        return audio.mean(dim=0) if audio.ndim > 1 else audio
    elif isinstance(audio, np.ndarray):
        return np.mean(audio, axis=0) if audio.ndim > 1 else audio
    else:
        raise ValueError("Audio data must be either a NumPy array or a PyTorch tensor.")


def pad_audio(audio, max_length, sample_rate, padding_value=0.0):
    """
    Pads an audio array to the specified length in seconds.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): Input audio array of shape (samples,).
        max_length (float): Desired maximum length in seconds.
        sample_rate (int): Sample rate of the audio.
        padding_value (float): Value used for padding. Default is 0.0 (silence).

    Returns:
        Union[np.ndarray, torch.Tensor]: Padded audio array.
    """
    target_length = int(max_length * sample_rate)
    current_length = audio.shape[-1]

    if current_length >= target_length:
        return audio

    padding_size = target_length - current_length

    if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
        return torch.cat([audio, torch.full((padding_size,), padding_value)], dim=-1)
    elif isinstance(audio, np.ndarray):
        return np.pad(audio, (0, padding_size), mode='constant', constant_values=padding_value)
    else:
        raise ValueError("Audio data must be either a NumPy array or a PyTorch tensor.")


def truncate_audio(audio, max_length, sample_rate):
    """
    Truncates an audio array to the specified length in seconds.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): Input audio array of shape (samples,).
        max_length (float): Desired maximum length in seconds.
        sample_rate (int): Sample rate of the audio.

    Returns:
        Union[np.ndarray, torch.Tensor]: Truncated audio array.
    """
    target_length = int(max_length * sample_rate)
    return audio[..., :target_length]  # Supports both np and torch slicing


def normalize(audio, method="min_max"):
    """
    Normalizes an audio array based on the specified method.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): Input audio array.
        method (str): Normalization method, one of "min_max", "standard", "peak", "rms". Default is "min_max".

    Returns:
        Union[np.ndarray, torch.Tensor]: Normalized audio array.
    
    Raises:
        ValueError: If an unsupported normalization method is provided.
    """
    if method == "min_max":
        min_val = audio.min()
        max_val = audio.max()
        normalized_audio = (audio - min_val) / (max_val - min_val + 1e-8)

    elif method == "standard":
        mean = audio.mean()
        std = audio.std()
        normalized_audio = (audio - mean) / (std + 1e-8)

    elif method == "peak":
        peak = abs(audio).max()
        normalized_audio = audio / (peak + 1e-8)

    elif method == "rms":
        rms = np.sqrt(np.mean(audio**2))
        normalized_audio = audio / (rms + 1e-8)

    else:
        raise ValueError("Unsupported normalization method. Choose from 'min_max', 'standard', 'peak', or 'rms'.")

    # Ensure the type of output matches the input (torch or numpy)
    if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
        return torch.tensor(normalized_audio, dtype=audio.dtype)
    else:
        return normalized_audio


def rms_normalize(audio, target_db=-20.0):
    """
    Normalizes an audio signal to a target RMS level.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): Input audio array.
        target_db (float): Target RMS level in dB. Default is -20 dB.

    Returns:
        Union[np.ndarray, torch.Tensor]: RMS-normalized audio array.
    """
    rms = np.sqrt(np.mean(audio**2))
    target_rms = 10 ** (target_db / 20)
    gain = target_rms / (rms + 1e-8)
    return audio * gain


if __name__ == "__main__":
    # Example usage for testing purposes
    from sappl.io import load_audio

    # Load an example audio file
    test_audio_path = "../samples/music_sample.wav"  # Replace with a valid path
    audio = load_audio(test_audio_path, sample_rate=16000, mono=True)

    # Convert to mono
    mono_audio = convert_to_mono(audio)
    print("Converted to mono, shape:", mono_audio.shape)

    # Pad audio to 5 seconds
    padded_audio = pad_audio(mono_audio, max_length=5.0, sample_rate=16000)
    print("Padded audio shape:", padded_audio.shape)

    # Truncate audio to 5 seconds
    truncated_audio = truncate_audio(padded_audio, max_length=5.0, sample_rate=16000)
    print("Truncated audio shape:", truncated_audio.shape)

    # Apply peak normalization
    peak_normalized_audio = normalize(mono_audio, method="peak")
    print("Peak-normalized audio min/max:", peak_normalized_audio.min(), peak_normalized_audio.max())

    # Apply RMS normalization
    rms_normalized_audio = rms_normalize(mono_audio, target_db=-20.0)
    print("RMS-normalized audio min/max:", rms_normalized_audio.min(), rms_normalized_audio.max())
