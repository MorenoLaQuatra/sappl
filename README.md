# 🎶 SAPPL: Simple Audio Pre-Processing Library

SAPPL (Simple Audio Pre-Processing Library) is a toolkit for audio preprocessing, designed specifically for deep learning applications such as speech classification, audio representation learning, and general audio preprocessing tasks. With SAPPL, you can easily load, transform, and extract features from audio files with minimal boilerplate code. Perfect for building model-ready audio datasets.

---

## 🔖 Table of Contents

- [🎶 SAPPL: Simple Audio Pre-Processing Library](#-sappl-simple-audio-pre-processing-library)
  - [🔖 Table of Contents](#-table-of-contents)
  - [Installation](#installation)
  - [📦 Implemented Modules](#-implemented-modules)
    - [1. `io.py` - Audio I/O Operations](#1-iopy---audio-io-operations)
      - [Functions](#functions)
      - [Example](#example)
    - [2. `transform.py` - Time-Frequency Transformations](#2-transformpy---time-frequency-transformations)
      - [Functions](#functions-1)
      - [Example](#example-1)
    - [3. `utils.py` - Utility Functions](#3-utilspy---utility-functions)
      - [Functions](#functions-2)
      - [Example](#example-2)
    - [4. `feature_extraction.py` - Feature Extraction](#4-feature_extractionpy---feature-extraction)
      - [Functions](#functions-3)
      - [Example](#example-3)
    - [5. `processor.py` - Audio Processor Class](#5-processorpy---audio-processor-class)
      - [Example](#example-4)
  - [🚀 Current Development](#-current-development)
  - [📄 License](#-license)
  - [❓ FAQ](#-faq)

---

## Installation

Install the package directly from PyPI using pip:

```bash
pip install sappl
```

or clone the repository and install it locally:

```bash
git clone https://github.com/MorenoLaQuatra/sappl
cd sappl
pip install .
```

---

## 📦 Implemented Modules

### 1. `io.py` - Audio I/O Operations

Handles loading and saving audio files with support for various file formats. Resamples audio to the specified sample rate and converts it to mono if required.

#### Functions

- **`load_audio(file_path, sample_rate=16000, mono=True)`**: Loads audio, resamples to the target rate, and optionally converts to mono.
- **`save_audio(file_path, audio, sample_rate=16000)`**: Saves audio data to a specified path and supports `.wav`, `.flac`, `.mp3`, and `.ogg` formats.

#### Example

```python
from sappl.io import load_audio, save_audio

audio_data = load_audio("path/to/file.wav", sample_rate=16000, mono=True)
print("Audio loaded:", audio_data.shape)

save_audio("path/to/save_file.wav", audio_data, sample_rate=16000)
print("Audio saved successfully.")
```

---

### 2. `transform.py` - Time-Frequency Transformations

Performs transformations like Short-Time Fourier Transform (STFT) and inverse STFT, as well as conversions between magnitude and phase. All outputs follow the `(T, F)` format, where `T` is time and `F` is frequency, making it ideal for deep learning.

#### Functions

- **`stft(audio, n_fft=2048, hop_length=512, win_length=None)`**: Computes the STFT and returns a `(T, F)` matrix.
- **`istft(stft_matrix, hop_length=512, win_length=None)`**: Reconstructs audio from an STFT matrix.
- **`magphase(stft_matrix)`**: Separates the magnitude and phase.
- **`compute_mel_spectrogram(audio, sample_rate=16000, n_fft=2048, hop_length=512, n_mels=128, f_min=0.0, f_max=None)`**: Computes the Mel spectrogram in dB scale.
- **`reconstruct_waveform(magnitude, phase, hop_length=512, win_length=None)`**: Reconstructs the waveform from magnitude and phase spectrograms.

#### Example

```python
from sappl.transform import stft, istft, magphase, compute_mel_spectrogram

# Compute STFT
stft_matrix = stft(audio_data)
print("STFT shape:", stft_matrix.shape)

# Separate magnitude and phase
magnitude, phase = magphase(stft_matrix)
print("Magnitude shape:", magnitude.shape)

# Compute Mel spectrogram
mel_spec = compute_mel_spectrogram(audio_data, sample_rate=16000)
print("Mel spectrogram shape:", mel_spec.shape)

# Reconstruct waveform from magnitude and phase
reconstructed_audio = reconstruct_waveform(magnitude, phase, hop_length=512)
print("Reconstructed audio shape:", reconstructed_audio.shape)
```

---

### 3. `utils.py` - Utility Functions

Provides utilities for handling audio data, such as padding, truncation, mono conversion, and normalization methods.

#### Functions

- **`convert_to_mono(audio)`**: Converts stereo audio to mono by averaging channels.
- **`pad_audio(audio, max_length, sample_rate, padding_value=0.0)`**: Pads audio to a specified duration.
- **`truncate_audio(audio, max_length, sample_rate)`**: Truncates audio to a specified duration.
- **`normalize(audio, method="min_max")`**: Normalizes audio using `min_max`, `standard`, `peak`, or `rms` methods.
- **`rms_normalize(audio, target_db=-20.0)`**: Normalizes audio to a target RMS level in dB.

#### Example

```python
from sappl.utils import convert_to_mono, pad_audio, truncate_audio, normalize, rms_normalize

# Convert to mono
mono_audio = convert_to_mono(audio_data)
print("Mono audio shape:", mono_audio.shape)

# Pad to 5 seconds
padded_audio = pad_audio(mono_audio, max_length=5.0, sample_rate=16000)
print("Padded audio shape:", padded_audio.shape)

# Normalize using peak normalization
peak_normalized = normalize(mono_audio, method="peak")
print("Peak-normalized min/max:", peak_normalized.min(), peak_normalized.max())

# RMS normalize to -20 dB
rms_normalized = rms_normalize(mono_audio, target_db=-20.0)
print("RMS-normalized min/max:", rms_normalized.min(), rms_normalized.max())
```

---

### 4. `feature_extraction.py` - Feature Extraction

Extracts key audio features useful for machine learning tasks, such as MFCC, Chroma, Tonnetz, Zero-Crossing Rate, and Spectral Contrast.

#### Functions

- **`extract_mfcc`**: Computes Mel-Frequency Cepstral Coefficients (MFCCs).
- **`extract_chroma`**: Extracts chroma features.
- **`extract_tonnetz`**: Extracts tonal centroid features.
- **`extract_zero_crossing_rate`**: Computes the zero-crossing rate.
- **`extract_spectral_contrast`**: Computes spectral contrast, capturing peak-valley differences in energy across frequency bands.

#### Example

```python
from sappl.feature_extraction import extract_mfcc, extract_chroma, extract_tonnetz, extract_zero_crossing_rate, extract_spectral_contrast

# Extract MFCCs
mfcc = extract_mfcc(audio_data, sample_rate=16000)
print("MFCC shape:", mfcc.shape)

# Extract Chroma features
chroma = extract_chroma(audio_data, sample_rate=16000)
print("Chroma shape:", chroma.shape)

# Extract Tonnetz features
tonnetz = extract_tonnetz(audio_data, sample_rate=16000)
print("Tonnetz shape:", tonnetz.shape)
```

---

### 5. `processor.py` - Audio Processor Class

The `AudioProcessor` class provides centralized access to all core functions in SAPPL, acting as an adaptable interface for loading, saving, transforming, and extracting features from audio files. It simplifies audio processing by allowing users to set processing parameters once at initialization (e.g., `n_fft`, `hop_length`, `n_mels`), reducing the need to pass them repeatedly for each function call.

#### Example

```python
from sappl.processor import AudioProcessor

# Initialize AudioProcessor with default parameters, setting them once
processor = AudioProcessor(sample_rate=16000, max_length=5.0, n_fft=1024, hop_length=256, n_mels=40)

# Load and preprocess audio
audio = processor.load_audio("path/to/file.wav")
audio_mono = processor.convert_to_mono(audio)
normalized_audio = processor.normalize(audio_mono, method="peak")

# Extract MFCC features
mfcc_features = processor.extract_mfcc(normalized_audio)
print("MFCC features shape:", mfcc_features.shape)

# Compute STFT and Mel Spectrogram
stft_matrix = processor.stft(normalized_audio)
mel_spectrogram = processor.compute_mel_spectrogram(normalized_audio)
print("Mel Spectrogram shape:", mel_spectrogram.shape)

# Reconstruct waveform from magnitude and phase
magnitude, phase = processor.magphase(stft_matrix)
reconstructed_audio = processor.reconstruct_waveform(magnitude, phase)
print("Reconstructed audio shape:", reconstructed_audio.shape)
```

---

## 🚀 Current Development

SAPPL is continuously expanding to include:
- **Augmentation**: Adding pitch shifting, time-stretching, and other augmentation utilities.
- **Filtering**: Implementing audio filters like band-pass and noise reduction for robust preprocessing.

---

## 📄 License

MIT License

---

SAPPL is built to streamline audio preprocessing for deep learning, with flexibility, consistency, and ease of use at its core. Happy coding! 🎉

---

## ❓ FAQ

<details>
<summary>1. Is SAPPL just a wrapper around librosa?</summary>

No, while SAPPL leverages `librosa` for many underlying functions, it provides additional features, modularity, and flexibility designed specifically for deep learning applications. SAPPL centralizes preprocessing functions, supports both NumPy and PyTorch arrays, enforces consistent `(T, F)` output for deep learning, and includes custom utilities such as padding, truncation, and normalization. SAPPL is more than a wrapper; it's a specialized framework for streamlined and robust audio processing.
</details>

<details>
<summary>2. Does SAPPL support PyTorch Tensors natively?</summary>

Yes, SAPPL is designed to handle both **NumPy arrays and PyTorch tensors**. When PyTorch tensors are passed, functions automatically handle them by converting to NumPy arrays internally, allowing compatibility without manual conversions. This flexibility allows SAPPL to fit seamlessly into PyTorch-based workflows.
</details>

<details>
<summary>3. Can I use custom functions with the AudioProcessor class?</summary>

Absolutely! The `AudioProcessor` class includes an `add_custom_function` method that allows you to add any callable function dynamically. This feature lets you expand the functionality of SAPPL by adding new processing steps or custom feature extractors without modifying the core library.
</details>

<details>
<summary>4. How can I contribute to SAPPL?</summary>

Contributions are welcome, especially as SAPPL continues to grow. To contribute, please fork the repository, create a new branch for your feature or bug fix, and submit a pull request. For substantial changes, feel free to open an issue first to discuss it. Please ensure your code is well-documented and tested.
</details>

<details>
<summary>5. What features are planned for future versions of SAPPL?</summary>

SAPPL aims to include:
- **Data Augmentation**: Options for pitch shifting, time-stretching, and adding noise to support robust training.
- **Filtering Options**: Basic filtering utilities such as band-pass filters and noise reduction techniques.
- **More Feature Extraction**: Additional features commonly used in speech and audio analysis.
  
Stay tuned and check our GitHub repository for updates on new releases!
</details>

---