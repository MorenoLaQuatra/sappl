import numpy as np
from sappl.transform import stft, istft, compute_mel_spectrogram, magphase, reconstruct_waveform
from sappl.io import load_audio

def test_stft_and_istft():
    audio_data = load_audio("samples/music_sample.wav", sample_rate=16000, mono=True)
    stft_matrix = stft(audio_data)
    reconstructed_audio = istft(stft_matrix)
    
    # Allow a tolerance of 1% length difference
    length_diff = abs(len(audio_data) - len(reconstructed_audio))
    assert length_diff <= 0.01 * len(audio_data), "Reconstructed audio length deviates more than 1% from original"

def test_compute_mel_spectrogram():
    audio_data = load_audio("samples/music_sample.wav", sample_rate=16000, mono=True)
    mel_spec = compute_mel_spectrogram(audio_data, sample_rate=16000)
    assert mel_spec is not None
    assert mel_spec.shape[1] == 128  # Check if mel bands are correct

def test_magphase():
    audio_data = load_audio("samples/music_sample.wav", sample_rate=16000, mono=True)
    stft_matrix = stft(audio_data)
    magnitude, phase = magphase(stft_matrix)
    assert magnitude.shape == phase.shape  # Magnitude and phase should have the same shape
    assert np.all(magnitude >= 0), "Magnitude should not contain negative values"

def test_reconstruct_waveform():
    audio_data = load_audio("samples/music_sample.wav", sample_rate=16000, mono=True)
    stft_matrix = stft(audio_data)
    magnitude, phase = magphase(stft_matrix)
    
    # Reconstruct waveform from magnitude and phase
    reconstructed_audio = reconstruct_waveform(magnitude, phase, hop_length=512)
    
    # Allow a tolerance of 1% length difference
    length_diff = abs(len(audio_data) - len(reconstructed_audio))
    assert length_diff <= 0.01 * len(audio_data), "Reconstructed audio length deviates more than 1% from original"