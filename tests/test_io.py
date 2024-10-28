import os
from sappl.io import load_audio, save_audio

def test_load_audio():
    audio_data = load_audio("samples/music_sample.wav", sample_rate=16000, mono=True)
    assert audio_data is not None
    assert audio_data.ndim == 1  # Mono audio should be 1D

def test_save_audio():
    audio_data = load_audio("samples/music_sample.wav", sample_rate=16000, mono=True)
    save_audio("samples/music_sample_saved.wav", audio_data, sample_rate=16000)
    assert os.path.exists("samples/music_sample_saved.wav")
    os.remove("samples/music_sample_saved.wav")  # Clean up after test
