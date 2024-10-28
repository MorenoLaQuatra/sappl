import numpy as np
from sappl.utils import pad_audio, truncate_audio, normalize, rms_normalize
from sappl.io import load_audio

def test_pad_audio():
    audio_data = load_audio("samples/music_sample.wav", sample_rate=16000, mono=True)
    padding_to = 16000 + len(audio_data)  # Pad to 1s + original length
    padded_audio = pad_audio(audio_data, padding_to / 16000, sample_rate=16000)
    assert len(padded_audio) == padding_to  # Check if padded length is correct

def test_truncate_audio():
    audio_data = load_audio("samples/music_sample.wav", sample_rate=16000, mono=True)
    truncated_audio = truncate_audio(audio_data, max_length=0.5, sample_rate=16000)
    assert len(truncated_audio) == 8000  # Check if truncated length is correct

def test_normalize():
    audio_data = load_audio("samples/music_sample.wav", sample_rate=16000, mono=True)
    norm_audio = normalize(audio_data, method="peak")
    assert np.max(np.abs(norm_audio)) <= 1  # Peak normalization should limit max to 1

def test_rms_normalize():
    audio_data = load_audio("samples/music_sample.wav", sample_rate=16000, mono=True)
    rms_audio = rms_normalize(audio_data, target_db=-20.0)
    assert rms_audio is not None
