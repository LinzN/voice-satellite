#!/usr/bin/env bash
# Train the "Hey Mirra" wake word. Run inside WSL2 (Ubuntu) on the Windows PC with the NVIDIA GPU:
#
#   bash /mnt/c/Users/<you>/source/mirra-wakeword/train.sh
#
# Everything heavy (Python environment, ~40 GB of data, checkpoints) lives in ~/mirra-wakeword inside WSL,
# because the Windows drive is slow from WSL. The finished model and the test results are copied back next
# to this script. The script can be stopped and started again; finished stages are skipped.
#
# Options are passed on to mww_train.py, e.g. --cpu, --steps 40000, --only train
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${MWW_WORK:-$HOME/mirra-wakeword}"
MWW_COMMIT=4665173cd35f1cff9a61e06fc427f124766c488e
PSG_COMMIT=2971426a55072f7d22fec416ca7800df8bd23207

mkdir -p "$WORK"
exec > >(tee -a "$HERE/train-output.log") 2>&1
echo "=== $(date) start, work folder $WORK"

# system tools
missing=()
for tool in git curl unzip tar; do command -v "$tool" >/dev/null || missing+=("$tool"); done
if ((${#missing[@]})); then
  echo "Installing ${missing[*]} (sudo password needed)"
  sudo apt-get update && sudo apt-get install -y "${missing[@]}"
fi

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  echo "WARNING: nvidia-smi not found in WSL - the GPU driver is missing, training will use the CPU"
fi

free_gb=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc 0-9)
if ((free_gb < 45)); then
  echo "WARNING: only ${free_gb} GB free in $WORK, about 45 GB are needed"
fi

# Python 3.10 environment with uv
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd "$WORK"
if [ ! -x venv/bin/python ]; then
  uv venv -p 3.10 venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

clone() { # url dir commit
  if [ ! -d "$2/.git" ]; then git clone -q "$1" "$2"; fi
  git -C "$2" fetch -q origin "$3" 2>/dev/null || true
  git -C "$2" checkout -q "$3"
}
clone https://github.com/kahrendt/microWakeWord microWakeWord "$MWW_COMMIT"
clone https://github.com/rhasspy/piper-sample-generator piper-sample-generator "$PSG_COMMIT"

if [ ! -f venv/.installed ]; then
  # PyTorch with CUDA 12.8 (needed for RTX 50xx / Blackwell) for the sample generator
  uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
  uv pip install "piper-tts==1.3.0" "audiomentations>=0.40" webrtcvad-wheels "numpy>=2,<3" "datasets==3.6.0" \
    soundfile librosa scipy tqdm pyyaml mmap_ninja pymicro-features ai-edge-litert tensorboard "tensorflow[and-cuda]>=2.18" \
    "git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f"
  uv pip install --no-deps -e ./microWakeWord
  touch venv/.installed
fi
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
deactivate

# (cusolver is pinned: newer wheels no longer ship libcusolver.so.11, which TensorFlow loads)
# Separate environment for the training: TensorFlow brings its own CUDA 12 libraries, which clash with the
# ones of PyTorch in the same environment (TensorFlow then does not find the GPU).
if [ ! -f venv-tf/.installed ]; then
  rm -rf venv-tf
  uv venv -p 3.10 venv-tf
  VIRTUAL_ENV="$WORK/venv-tf" uv pip install "tensorflow[and-cuda]>=2.18" "nvidia-cusolver-cu12==11.6.3.83" tensorboard ai-edge-litert "numpy>=2,<3" \
    "datasets==3.6.0" "audiomentations>=0.40" soundfile scipy tqdm pyyaml mmap_ninja pymicro-features webrtcvad-wheels \
    "git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f"
  VIRTUAL_ENV="$WORK/venv-tf" uv pip install --no-deps -e ./microWakeWord
  touch venv-tf/.installed
fi
venv-tf/bin/python -c "import tensorflow as tf; print('TensorFlow', tf.__version__, 'GPU', tf.config.list_physical_devices('GPU'))" 2>&1 \
  | grep -E "TensorFlow|Could not|Cannot|dlopen|\.so" || true

venv/bin/python "$HERE/mww_train.py" --work "$WORK" --out "$HERE" --train-python "$WORK/venv-tf/bin/python" "$@"
echo "=== $(date) done. Model: $HERE/hey_mirra.tflite"
