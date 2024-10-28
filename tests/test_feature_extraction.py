from sappl.feature_extraction import extract_mfcc, extract_chroma, extract_tonnetz, extract_zero_crossing_rate, extract_spectral_contrast
from sappl.io import load_audio

def test_extract_mfcc():
    audio_data = load_audio("samples/music_sample.wav", sample_rate=16000, mono=True)
    mfcc = extract_mfcc(audio_data, sample_rate=16000)
    assert mfcc.shape[1] == 13  # Should return 13 MFCCs

def test_extract_chroma():
    audio_data = load_audio("samples/music_sample.wav", sample_rate=16000, mono=True)
    chroma = extract_chroma(audio_data, sample_rate=16000)
    assert chroma.shape[1] == 12  # Should return 12 chroma bins

def test_extract_tonnetz():
    audio_data = load_audio("samples/music_sample.wav", sample_rate=16000, mono=True)
    tonnetz = extract_tonnetz(audio_data, sample_rate=16000)
    assert tonnetz.shape[1] == 6  # Should return 6 tonal centroid features
