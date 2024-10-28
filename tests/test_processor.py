from sappl.processor import AudioProcessor

def test_audio_processor():
    processor = AudioProcessor(sample_rate=16000, max_length=1.0)

    # Load and preprocess audio
    audio = processor.load_audio("samples/music_sample.wav")
    assert audio is not None

    # Convert to mono, normalize, and pad
    mono_audio = processor.convert_to_mono(audio)
    assert mono_audio.ndim == 1
    padded_audio = processor.pad_audio(mono_audio)
    assert len(padded_audio) >= 16000
    truncated_audio = processor.truncate_audio(mono_audio)
    assert len(truncated_audio) <= 16000

    # Extract features
    mfcc = processor.extract_mfcc(padded_audio)
    assert mfcc.shape[1] == 13
    chroma = processor.extract_chroma(padded_audio)
    assert chroma.shape[1] == 12
