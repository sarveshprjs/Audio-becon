# Getting Started

Follow these steps to get up and running with **Audio-becon**:

## Requirements

**Hardware**
- Minimum 4 GB RAM (8 GB+ recommended)
- CPU or NVIDIA GPU (recommended for faster training)
- Microphone or .wav files for inference/testing

**Operating System**
- Ubuntu 18.04+, macOS, or Windows 10/11

**Python**
- Python 3.7 or higher

**Dependencies**
- See `requirements.txt` in this repository, or install manually:
  - numpy
  - librosa
  - keras
  - tensorflow or tensorflow-gpu
  - scikit-learn
  - matplotlib

## Installation

**1. Clone the Repository**
git clone https://github.com/sarveshprjs/Audio-becon.git
cd Audio-becon



**2. Create and activate a virtual environment (recommended)**
python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate



**3. Install dependencies**
pip install -r requirements.txt


*(Or, manually: `pip install numpy librosa keras tensorflow scikit-learn matplotlib`)*

## Dataset Preparation

Download the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html) and place it in the recommended directory (see code comments or documentation for guidance).

## Sample Commands

**Train the model:**
python train.py --data_dir path/to/UrbanSound8K


**Classify a single audio file:**

python predict.py --file path/to/example.wav


**Run a demo (if provided):**

python demo.py


## Troubleshooting

- Ensure all dependencies are installed for your Python version.
- For GPU acceleration, confirm CUDA/cuDNN are correctly set up ([TensorFlow GPU guide](https://www.tensorflow.org/install/gpu)).
- If you encounter `ModuleNotFoundError`, double-check your virtual environment.

---

**You are now ready to explore environmental sound classification with Audio-becon!**  
See the `docs/` folder for more details and advanced usage.
