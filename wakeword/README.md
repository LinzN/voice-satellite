# "Hey Mirra" wake word training

Trains the wake word for the Mirra voice satellites with [microWakeWord](https://github.com/kahrendt/microWakeWord)
on the local NVIDIA GPU, inside WSL2.

## Run

Open the Ubuntu (WSL) terminal and start:

```sh
bash /mnt/c/Users/nikla/source/mirra-wakeword/train.sh
```

- The first run installs Python 3.10, PyTorch, TensorFlow and microWakeWord into `~/mirra-wakeword/venv`
  (asks for the sudo password only if `git`, `curl`, `unzip` or `tar` are missing).
- Needs about **45 GB** free space in WSL and a few hours. Downloads about 15 GB.
- The script can be stopped (Ctrl+C) and started again; finished stages are skipped.
- Everything is logged to `train-output.log` next to the script.

When it is done, this folder contains:

| File | Content |
|---|---|
| `hey_mirra.tflite` | the quantized streaming model for the satellites |
| `hey_mirra_model.h` | the model as C array for the firmware (goes next to `mirra-voice.ino`) |
| `hey_mirra.json` | model manifest (ESPHome format), the cutoff is tuned afterwards from the test results |
| `results/` | test results (false accepts per hour, recall), training log and parameters |

## Your own recordings

Real recordings of the people in the house help the most. Record a voice memo (phone or PC, any format ffmpeg
reads: m4a, mp3, wav, ogg, ...) per person: say "Hey Mirra" 50-100 times with a pause of 1-2 seconds, at
different distances, louder and quieter, faster and slower, the way you would really say it. Put the files into
`recordings/` next to this script and start `train.sh` again.

The script cuts the memos into single utterances (`recordings/segments/`, listen to a few to check) and uses
each of them many times with different background noise and room echo. New or changed recordings are picked up
automatically on the next run; only the features and the training are repeated.

## Stages

1. **samples** – about 36,000 synthetic "Hey Mirra" samples (all German Piper voices, 236 speakers in the
   largest one, and 500 mixed English voices, with different speed and variation, also "Mirraa" / "Mira") and 15,000 samples of look-alike phrases ("Mirra" alone, "Hey Mia",
   "Hey Kira", "Hey Siri", "mir da", ...) that must *not* wake the speaker.
2. **data** – room impulse responses, AudioSet and FMA background noise/music, and the negative feature sets of
   microWakeWord (speech, dinner party, no speech) from HuggingFace.
3. **features** – augmentation (noise, echo, EQ, distortion) and spectrograms.
4. **train** – 30,000 steps of the MixedNet model, then conversion to a quantized streaming TFLite model and a
   test on held-out data. Uses the GPU if TensorFlow can use it, otherwise the CPU (slower).
5. **export** – copies the results here.

## Options

```sh
train.sh --cpu              # train on the CPU
train.sh --steps 50000      # longer training
train.sh --only train       # repeat one stage (e.g. after changing parameters)
MWW_WORK=/data/mww train.sh # other work folder
```

To start over completely, delete `~/mirra-wakeword` in WSL.

## Notes

- On RTX 50xx (Blackwell) TensorFlow may compile its GPU kernels on the first start, which can take a while.
  If the GPU test fails, training falls back to the CPU automatically.
- The background datasets have mixed licenses; the resulting model is for personal, non-commercial use.
