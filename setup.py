from setuptools import setup, find_packages

# Load the README file for the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sappl",
    version="0.1.0",
    author="Moreno La Quatra",
    author_email="moreno.laquatra@gmail.com",
    description="Simple Audio Pre-Processing Library for deep learning audio applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MorenoLaQuatra/sappl",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "librosa",
        "soundfile",
        "pydub",
        "torch",
        "ffmpeg-python",
    ],
)
