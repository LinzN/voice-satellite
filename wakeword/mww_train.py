#!/usr/bin/env python3
"""
Train the "Hey Mirra" wake word for the Mirra voice satellites with microWakeWord.

Stages (each one is skipped when its marker file exists, so the script can be restarted):
  samples   synthetic wake word and look-alike phrases with German and English Piper generators
  data      download room impulse responses, background noise and the negative feature sets
  features  augment the samples and compute the spectrogram features
  train     train the model and convert it to a quantized streaming TFLite model
  export    copy the model, the test results and a manifest into the output folder

Usage: python mww_train.py --work ~/mirra-wakeword --out /mnt/c/.../mirra-wakeword [--smoke]
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

POSITIVE_PHRASES = ["Hey Mirra", "Hey Mirra!", "Hey, Mirra.", "Hei Mirra", "Hey Mirra?", "hey mirra"]
# How people in the house actually say it: short i, stressed r and a long a ("Mirrraa"), sometimes a long i
POSITIVE_VARIANTS_DE = ["Hey Mirraa", "Hey Mirrah!", "Hey, Mirrraa.", "Hey Mira", "Hey Mira!", "Hei Mirraa", "Hey Mihra"]
POSITIVE_VARIANTS_EN = ["Hey Mirrah", "Hey Mira", "Hey Meerah", "Hey Mirra!"]

# Phrases that sound close to the wake word or are said often in the house. The model learns to stay quiet
# for them; "Mirra" alone is included on purpose, so talking *about* Mirra does not wake her.
ADVERSARIAL_DE = [
    "Mirra", "Mirra, hörst du?", "Mira", "Hey", "Hey du", "Hey Mia", "Hey Emma", "Hey Lina", "Hey Pia",
    "Hey Marie", "Hey Miriam", "Hey Kira", "Hey Lara", "Hey Klara", "Hey Ida", "Hey Vera", "Hey Merle",
    "Hey Mama", "Hey Mila", "Hey Nina", "Hey Tim", "Hey Siri", "Alexa", "Hey Google", "hey mir da",
    "hey mit dir", "hier da", "mir da", "Kamera", "Hey Leute", "Hey Jungs", "hey wie geht's", "ihr da",
    "Hey Pierre", "Hey Hilda", "Hey Mira, nein", "Himmel", "Mitra", "Herr Mirr", "ey Bruder", "Hey Miri",
]
ADVERSARIAL_EN = [
    "Mirra", "Hey", "Hey Siri", "Hey Google", "Alexa", "Hey Mia", "Hey Kira", "Mirror", "Hey Mister",
    "Hey Nira", "Hey Ira", "here we go", "hey there", "Hey Emma", "Hey Mary", "hey me", "Hey Meera",
]

# Look-alikes that turned out to be too close (or to contain the wake word): the samples are generated with the
# full lists above (the file number decides the phrase) and these are removed afterwards.
EXCLUDED_ADVERSARIAL = {"Hey Mira, nein", "Hey Miri", "Hey Meera"}
MODEL_VERSION = "v3"

HF = "https://huggingface.co/datasets"
NEGATIVE_FEATURES = ["dinner_party.zip", "dinner_party_eval.zip", "no_speech.zip", "speech.zip"]
# English: the multi-speaker generator of piper-sample-generator (mixes speakers).
# German: the regular Piper voices. The German generator (de_DE-mls-medium.pt) produces 3-8 s of babble for
# "Hey Mirra" instead of ~0.7 s, which ruined the first model.
GENERATORS = {
    "en": "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt",
}
GERMAN_VOICE_SHARE = {}  # optional larger share for single voices, the others split the rest
# de_DE-mls-medium is trained on the same data as the German generator and has the same problem:
# median 3 s, up to 14 s for "Hey Mirra"
EXCLUDED_VOICES = ("mls",)


def log(message):
    print(f"\n[{time.strftime('%H:%M:%S')}] === {message}", flush=True)


def run(command, **kwargs):
    print("+ " + " ".join(str(c) for c in command), flush=True)
    subprocess.run([str(c) for c in command], check=True, **kwargs)


def done(work, stage):
    return (work / f".done-{stage}").exists()


def mark(work, stage):
    (work / f".done-{stage}").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def download(url, target):
    if target.exists() and target.stat().st_size > 0 and not (target.parent / (target.name + ".part")).exists():
        return
    part = target.parent / (target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", part, url])
    part.rename(target)


# --------------------------------------------------------------------------- samples

def generate(work, args, model, phrases, count, out_dir, max_speakers):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.wav")))
    if existing >= count:
        print(f"{out_dir}: {existing} samples present")
        return
    text_file = out_dir.parent / (out_dir.name + ".txt")
    text_file.write_text("\n".join(phrases) + "\n", encoding="utf-8")
    # PYTORCH_JIT=0: the generator's TorchScript modules re-profile for every new input shape, which makes
    # generation several times slower and fills the GPU memory; plain eager mode is much faster here
    env = dict(os.environ, PYTHONPATH=str(work / "piper-sample-generator"), PYTORCH_JIT="0")
    command = [sys.executable, "-m", "piper_sample_generator", text_file,
               "--model", model, "--max-samples", count, "--batch-size", args.batch_size,
               "--length-scales", "0.7", "0.85", "1.0", "1.15", "1.3",
               "--noise-scales", "0.4", "0.667", "0.9",
               "--noise-scale-ws", "0.5", "0.8", "1.0",
               "--slerp-weights", "0.0", "0.3", "0.5", "0.7", "1.0",
               "--output-dir", out_dir]
    if max_speakers:
        command += ["--max-speakers", max_speakers]
    run(command, env=env)


def prune_adversarial(work):
    """Remove samples of excluded look-alike phrases (file i was generated from phrase i % len(phrases))."""
    removed = 0
    for lang, phrases in (("de", ADVERSARIAL_DE), ("en", ADVERSARIAL_EN)):
        folder = work / "samples" / "adversarial" / lang
        bad = {i for i, phrase in enumerate(phrases) if phrase in EXCLUDED_ADVERSARIAL}
        for wav in folder.glob("*.wav"):
            if wav.stem.isdigit() and int(wav.stem) % len(phrases) in bad:
                wav.unlink()
                removed += 1
    return removed


def migrate(work):
    """Bring the work folder of an earlier run up to the current model version without regenerating everything."""
    marker = work / f".model-{MODEL_VERSION}"
    if marker.exists() or not (work / "samples" / "adversarial").exists():
        return
    log(f"Updating the work folder to model {MODEL_VERSION}")
    if not (work / ".model-v2").exists():
        print(f"removed {prune_adversarial(work)} samples of excluded look-alike phrases")
        shutil.rmtree(work / "features" / "adversarial", ignore_errors=True)
    # v3: German samples come from the Piper voices now (the German generator produced babble), plus
    # pronunciation variants; the broken German samples and all synthetic features are rebuilt
    for old in ("positive/de", "positive/de_variants", "adversarial/de"):
        shutil.rmtree(work / "samples" / old, ignore_errors=True)
    shutil.rmtree(work / "features" / "positive", ignore_errors=True)
    shutil.rmtree(work / "features" / "adversarial", ignore_errors=True)
    for stage in ("samples", "features", "train", "export"):
        (work / f".done-{stage}").unlink(missing_ok=True)
    marker.write_text("ok")


def german_voices(work, args):
    """Download all German Piper voices (ONNX)."""
    voices_dir = work / "voices"
    if args.smoke:
        return sorted(voices_dir.glob("*.onnx"))
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files("rhasspy/piper-voices")
    names = sorted(f for f in files if f.startswith("de/de_DE/") and f.endswith(".onnx"))
    if not names:
        raise SystemExit("No German voices found in rhasspy/piper-voices")
    voices = []
    for name in names:
        for suffix in ("", ".json"):
            hf_hub_download("rhasspy/piper-voices", name + suffix, local_dir=voices_dir)
        voices.append(voices_dir / name)
    return voices


def generate_german(work, args, phrases, total, out_root):
    """Generate with every German voice in parallel processes, the multi-speaker voice gets the largest share."""
    from concurrent.futures import ThreadPoolExecutor
    voices = [v for v in german_voices(work, args) if not any(key in v.stem for key in EXCLUDED_VOICES)]
    for folder in out_root.glob("*"):
        if folder.is_dir() and any(key in folder.name for key in EXCLUDED_VOICES):
            shutil.rmtree(folder)
    print("German voices: " + ", ".join(v.stem for v in voices), flush=True)
    big = [v for v in voices if any(key in v.stem for key in GERMAN_VOICE_SHARE)]
    rest = [v for v in voices if v not in big]
    big_share = sum(GERMAN_VOICE_SHARE.values()) if big and rest else (1.0 if big else 0.0)
    counts = {v: int(total * big_share / len(big)) for v in big}
    counts.update({v: int(total * (1 - big_share) / len(rest)) for v in rest})
    text_file = out_root.parent / (out_root.name + ".txt")
    out_root.mkdir(parents=True, exist_ok=True)
    text_file.write_text("\n".join(phrases) + "\n", encoding="utf-8")
    # ONNX voices run on the CPU; CUDA_VISIBLE_DEVICES keeps piper from asking for the CUDA provider
    env = dict(os.environ, PYTHONPATH=str(work / "piper-sample-generator"), CUDA_VISIBLE_DEVICES="")

    def run_voice(voice):
        target = out_root / voice.stem
        count = counts[voice]
        if target.exists() and len(list(target.glob("*.wav"))) >= count:
            return
        target.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "piper_sample_generator", str(text_file), "--model", str(voice),
                   "--max-samples", str(count), "--length-scales", "0.8", "0.9", "1.0", "1.1", "1.25",
                   "--noise-scales", "0.5", "0.667", "0.9", "--noise-scale-ws", "0.6", "0.8", "1.0",
                   "--output-dir", str(target)]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{voice.stem}: {result.stderr[-1500:]}")
        print(f"  {voice.stem}: {count} samples", flush=True)

    workers = max(1, min(len(voices), (os.cpu_count() or 2) // 2))
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run_voice, voices))


def stage_samples(work, args):
    models = work / "generators"
    for lang, url in GENERATORS.items():
        name = url.rsplit("/", 1)[1]
        download(url, models / name)
        source_json = work / "piper-sample-generator" / "models" / (name + ".json")
        shutil.copy(source_json, models / (name + ".json"))
    en = models / "en_US-libritts_r-medium.pt"
    german_positive = POSITIVE_PHRASES + POSITIVE_VARIANTS_DE
    german_adversarial = [p for p in ADVERSARIAL_DE if p not in EXCLUDED_ADVERSARIAL]
    generate_german(work, args, german_positive, args.positive_de + args.variants_de,
                    work / "samples" / "positive" / "de_voices")
    generate(work, args, en, POSITIVE_PHRASES, args.positive_en, work / "samples" / "positive" / "en", 500)
    generate(work, args, en, POSITIVE_VARIANTS_EN, args.variants_en, work / "samples" / "positive" / "en_variants", 500)
    generate_german(work, args, german_adversarial, args.adversarial_de, work / "samples" / "adversarial" / "de_voices")
    generate(work, args, en, ADVERSARIAL_EN, args.adversarial_en, work / "samples" / "adversarial" / "en", 500)
    prune_adversarial(work)
    (work / f".model-{MODEL_VERSION}").write_text("ok")


# --------------------------------------------------------------------------- data

def write_wav(path, samples, rate=16000):
    import numpy as np
    import scipy.io.wavfile
    scipy.io.wavfile.write(path, rate, (np.clip(samples, -1, 1) * 32767).astype(np.int16))


def convert_folder(source_glob_root, pattern, target):
    """Convert audio files to 16 kHz mono WAV."""
    import numpy as np
    import soundfile
    from scipy.signal import resample_poly
    from math import gcd
    from tqdm import tqdm
    target.mkdir(parents=True, exist_ok=True)
    files = list(Path(source_glob_root).glob(pattern))
    for file in tqdm(files, desc=f"converting {target.name}"):
        out = target / (file.stem + ".wav")
        if out.exists():
            continue
        try:
            audio, rate = soundfile.read(str(file), dtype="float32", always_2d=True)
        except Exception as error:  # broken or unsupported file: skip it
            print(f"skip {file.name}: {error}")
            continue
        audio = audio.mean(axis=1)
        if rate != 16000:
            divisor = gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor)
        write_wav(out, audio)


def audioset_background(data, parts):
    """Download the first balanced-train parts of AudioSet (tar with FLAC, or parquet) as 16 kHz WAV."""
    import io
    import re
    import numpy as np
    import soundfile
    from huggingface_hub import HfApi, hf_hub_download
    from scipy.signal import resample_poly
    from math import gcd

    target = data / "background_16k"
    target.mkdir(parents=True, exist_ok=True)
    stamp = data / "audioset.done"
    if stamp.exists():
        return
    files = HfApi().list_repo_files("agkphysics/AudioSet", repo_type="dataset")
    candidates = sorted(f for f in files if re.search(r"bal_train", f) and f.endswith((".tar", ".parquet")))
    if not candidates:
        raise RuntimeError("no balanced-train files in agkphysics/AudioSet: " + ", ".join(files[:20]))
    for name in candidates[:parts]:
        print(f"AudioSet: {name}", flush=True)
        local = Path(hf_hub_download("agkphysics/AudioSet", name, repo_type="dataset", local_dir=data / "audioset"))
        if name.endswith(".tar"):
            run(["tar", "-xf", local, "-C", data / "audioset"])
            convert_folder(data / "audioset", "**/*.flac", target)
            continue
        import pyarrow.parquet as pq
        table = pq.read_table(local)
        column = next(c for c in table.column_names if "audio" in c.lower() or "flac" in c.lower())
        for index, row in enumerate(table.column(column).to_pylist()):
            payload = row.get("bytes") if isinstance(row, dict) else row
            if not payload:
                continue
            try:
                audio, rate = soundfile.read(io.BytesIO(payload), dtype="float32", always_2d=True)
            except Exception:
                continue
            audio = audio.mean(axis=1)
            if rate != 16000:
                divisor = gcd(rate, 16000)
                audio = resample_poly(audio, 16000 // divisor, rate // divisor)
            write_wav(target / f"{Path(name).stem}_{index}.wav", audio)
    stamp.write_text("ok")


def stage_data(work, args):
    import numpy as np
    data = work / "data"
    data.mkdir(exist_ok=True)
    if args.smoke:
        # offline stand-ins: colored noise as background, decaying noise bursts as room impulses
        rng = np.random.default_rng(1)
        for folder in ("background_16k", "mit_rirs"):
            (data / folder).mkdir(exist_ok=True)
        for i in range(20):
            noise = np.cumsum(rng.normal(0, 1, 16000 * 10)) * 0.001
            noise = noise - np.convolve(noise, np.ones(400) / 400, mode="same")
            write_wav(data / "background_16k" / f"noise{i}.wav", noise * 0.3 + rng.normal(0, 0.02, noise.size))
            impulse = rng.normal(0, 1, 4000) * np.exp(-np.arange(4000) / (300 + 100 * i))
            write_wav(data / "mit_rirs" / f"rir{i}.wav", impulse / np.abs(impulse).max())
        return

    # room impulse responses
    rir = data / "mit_rirs"
    if not rir.exists() or not any(rir.iterdir()):
        import datasets
        rir.mkdir(exist_ok=True)
        dataset = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
        for row in dataset:
            name = row["audio"]["path"].split("/")[-1]
            write_wav(rir / name, np.asarray(row["audio"]["array"]), 16000)

    # background noise: a few AudioSet parts and the FMA music set. Dataset layouts on HuggingFace change,
    # so the files are looked up instead of hard coded, and a missing source is skipped with a warning.
    sources = 0
    try:
        audioset_background(data, args.audioset_parts)
        sources += 1
    except Exception as error:
        print(f"WARNING: AudioSet background noise not available: {error}")
    try:
        fma = data / "fma" / "fma_xs.zip"
        download(f"{HF}/mchl914/fma_xsmall/resolve/main/fma_xs.zip", fma)
        if not (data / "fma" / ".extracted").exists():
            run(["unzip", "-q", "-o", fma, "-d", data / "fma"])
            (data / "fma" / ".extracted").write_text("ok")
        convert_folder(data / "fma", "**/*.mp3", data / "fma_16k")
        sources += 1
    except Exception as error:
        print(f"WARNING: FMA music not available: {error}")
    if sources == 0:
        raise SystemExit("No background noise could be downloaded, check the internet connection / HuggingFace")

    # negative spectrogram features prepared for microWakeWord
    negatives = work / "negative_datasets"
    from huggingface_hub import HfApi, hf_hub_download
    available = HfApi().list_repo_files("kahrendt/microwakeword", repo_type="dataset")
    for name in NEGATIVE_FEATURES:
        if name not in available:
            raise SystemExit(f"{name} is missing in kahrendt/microwakeword, files there: {available}")
        archive = Path(hf_hub_download("kahrendt/microwakeword", name, repo_type="dataset", local_dir=negatives))
        stamp = negatives / (name + ".extracted")
        if not stamp.exists():
            run(["unzip", "-q", "-o", archive, "-d", negatives])
            stamp.write_text("ok")


# --------------------------------------------------------------------------- real recordings

def stage_recordings(work, out):
    """
    Cut real recordings (phone voice memos with "Hey Mirra" said many times, any format ffmpeg reads) from
    <out>/recordings into single utterances. The cut pieces are also written to recordings/segments, so they
    can be checked by listening.
    """
    import numpy as np
    import scipy.io.wavfile

    source = out / "recordings"
    target = work / "samples" / "real"
    if not source.exists():
        source.mkdir(parents=True, exist_ok=True)
    files = [f for f in source.iterdir() if f.is_file() and not f.name.startswith(".")]
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    check = source / "segments"
    shutil.rmtree(check, ignore_errors=True)
    if not files:
        print("No recordings in " + str(source) + " - training with synthetic voices only")
        return 0
    check.mkdir()
    total = 0
    for file in sorted(files):
        wav = work / "recording.wav"
        result = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(file), "-ac", "1", "-ar", "16000",
                                 "-sample_fmt", "s16", str(wav)])
        if result.returncode != 0:
            print(f"skip {file.name}: ffmpeg could not read it")
            continue
        _, audio = scipy.io.wavfile.read(wav)
        audio = audio.astype(np.float32) / 32768.0
        frame = 480  # 30 ms
        frames = len(audio) // frame
        if frames < 10:
            continue
        rms = np.sqrt(np.mean(audio[:frames * frame].reshape(frames, frame) ** 2, axis=1)) + 1e-6
        # relative thresholds: phone memos are often quiet, and noise suppression makes the pauses digital silence
        noise = np.percentile(rms, 20)
        loud = rms > max(noise * 4, np.percentile(rms, 95) * 0.15, 0.0003)
        # merge loud frames into utterances; pauses shorter than 0.35 s stay inside one utterance
        segments, start, quiet = [], None, 0
        for index, is_loud in enumerate(loud):
            if is_loud:
                if start is None:
                    start = index
                quiet = 0
            elif start is not None:
                quiet += 1
                if quiet > 12:
                    segments.append((start, index - quiet + 1))
                    start, quiet = None, 0
        if start is not None:
            segments.append((start, frames))
        count = 0
        for first, last in segments:
            length = (last - first) * 0.03
            if length < 0.35 or length > 2.5:
                continue
            a = max(0, first * frame - 2400)
            b = min(len(audio), last * frame + 2400)
            piece = audio[a:b]
            piece = piece / max(np.abs(piece).max(), 1e-4) * 0.5  # same level for every clip, augmentation varies it
            clip = (piece * 32767).astype(np.int16)
            name = f"{file.stem}_{count:03d}.wav"
            scipy.io.wavfile.write(target / name, 16000, clip)
            scipy.io.wavfile.write(check / name, 16000, clip)
            count += 1
        print(f"{file.name}: {count} utterances")
        total += count
        wav.unlink(missing_ok=True)
    print(f"{total} real utterances in total")
    return total


def recordings_changed(work, out):
    """New or changed recordings invalidate the features and the model."""
    source = out / "recordings"
    files = sorted(f"{f.name}:{f.stat().st_size}" for f in source.iterdir() if f.is_file()) if source.exists() else []
    signature = "segmenter-2\n" + "\n".join(files)
    stamp = work / ".recordings"
    if stamp.exists() and stamp.read_text() == signature:
        return False
    stamp.write_text(signature)
    for stage in ("features", "train", "export"):
        (work / f".done-{stage}").unlink(missing_ok=True)
    return True


# --------------------------------------------------------------------------- features

def make_features(clips_dir, out_dir, name, background, rirs, train_repeat=2):
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap

    clips = Clips(input_directory=str(clips_dir), file_pattern="**/*.wav", max_clip_duration_s=None,
                  remove_silence=False, random_split_seed=10, split_count=0.1)
    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1, "TanhDistortion": 0.1, "PitchShift": 0.1, "BandStopFilter": 0.1,
            "AddColorNoise": 0.1, "AddBackgroundNoise": 0.75, "Gain": 1.0, "RIR": 0.5,
        },
        impulse_paths=[str(rirs)],
        background_paths=[str(b) for b in background],
        background_min_snr_db=-5, background_max_snr_db=10,
        min_jitter_s=0.195, max_jitter_s=0.205,
    )
    for split, split_name, repeat, slide in (("training", "train", train_repeat, 10), ("validation", "validation", 1, 10),
                                             ("testing", "test", 1, 1)):
        target = out_dir / split / f"{name}_mmap"
        done_marker = out_dir / split / f".{name}.complete"
        if target.exists() and done_marker.exists():
            continue
        # an interrupted run leaves a half written feature set behind
        shutil.rmtree(target, ignore_errors=True)
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        spectrograms = SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=slide, step_ms=10)
        RaggedMmap.from_generator(out_dir=str(target),
                                  sample_generator=spectrograms.spectrogram_generator(split=split_name, repeat=repeat),
                                  batch_size=100, verbose=True)
        done_marker.write_text("ok")


def smoke_negatives(work):
    """Offline stand-in for the downloaded negative feature sets: features of noise clips."""
    import numpy as np
    from mmap_ninja.ragged import RaggedMmap
    from pymicro_features import MicroFrontend  # noqa: F401  (ensures the frontend is installed)
    from microwakeword.audio.audio_utils import generate_features_for_clip
    rng = np.random.default_rng(2)
    negatives = work / "negative_datasets"

    def clips(n, seconds):
        for _ in range(n):
            audio = rng.normal(0, 0.05, 16000 * seconds)
            yield generate_features_for_clip((audio * 32767).astype(np.int16), step_ms=10)

    for name in ("speech", "dinner_party", "no_speech"):
        for split in ("training", "validation", "testing"):
            target = negatives / name / split / "smoke_mmap"
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                RaggedMmap.from_generator(out_dir=str(target), sample_generator=clips(40, 3), batch_size=20)
    for split in ("validation_ambient", "testing_ambient"):
        target = negatives / "dinner_party_eval" / split / "smoke_mmap"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            RaggedMmap.from_generator(out_dir=str(target), sample_generator=clips(3, 60), batch_size=3)


def stage_features(work, args):
    data = work / "data"
    background = [p for p in (data / "background_16k", data / "fma_16k") if p.exists()]
    real = work / "samples" / "real"
    shutil.rmtree(work / "features" / "real", ignore_errors=True)
    if real.exists() and len(list(real.glob("*.wav"))) >= 10:
        # few real clips: each one is used many times with different noise and room echo
        make_features(real, work / "features" / "real", "real", background, data / "mit_rirs", train_repeat=40)
    stage_features_synthetic(work, args)


def stage_features_synthetic(work, args):
    data = work / "data"
    background = [p for p in (data / "background_16k", data / "fma_16k") if p.exists()]
    make_features(work / "samples" / "positive", work / "features" / "positive", "wakeword", background, data / "mit_rirs")
    make_features(work / "samples" / "adversarial", work / "features" / "adversarial", "adversarial", background,
                  data / "mit_rirs")
    if args.smoke:
        smoke_negatives(work)


# --------------------------------------------------------------------------- train

TRAIN_ENV = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="1",
                 # RTX 50xx: TensorFlow compiles its kernels from PTX on the first start; keep them cached
                 CUDA_CACHE_MAXSIZE="4294967296")


def gpu_usable(python):
    test = ("import tensorflow as tf\n"
            "g = tf.config.list_physical_devices('GPU')\n"
            "assert g, 'no GPU'\n"
            "x = tf.random.normal([1, 49, 40, 1])\n"
            "y = tf.keras.layers.Conv2D(8, 3)(x)\n"
            "print('GPU ok', g, float(tf.reduce_sum(y)))\n")
    try:
        log("Checking TensorFlow on the GPU (the first start can take a while)")
        result = subprocess.run([python, "-c", test], capture_output=True, text=True, timeout=3600,
                                env=dict(TRAIN_ENV, TF_CPP_MIN_LOG_LEVEL="0"))
        print(result.stdout[-2000:], result.stderr[-2000:])
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def dir_size(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def subset_negatives(work, budget_gb):
    """
    The training reads random spectrograms from all feature sets. When they do not fit into the RAM, every
    step waits for the disk. The large negative training sets are therefore reduced to a random subset that
    fits into the given budget; validation and test sets are used as they are.
    """
    import numpy as np
    from mmap_ninja.ragged import RaggedMmap
    source_root = work / "negative_datasets"
    target_root = work / "negative_subsets"
    shares = {"speech": 0.4, "dinner_party": 0.4, "no_speech": 0.2}
    rng = np.random.default_rng(3)
    for name, share in shares.items():
        source = source_root / name
        target = target_root / name
        for split in source.iterdir():
            if not split.is_dir():
                continue
            link = target / split.name
            if split.name != "training":
                if not link.exists():
                    link.parent.mkdir(parents=True, exist_ok=True)
                    link.symlink_to(split, target_is_directory=True)
                continue
            limit = budget_gb * share * 1024 ** 3
            total = dir_size(split)
            stamp = target / f".training-{budget_gb}"
            if stamp.exists():
                continue
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                shutil.rmtree(link)
            if total <= limit:
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(split, target_is_directory=True)
            else:
                fraction = limit / total
                print(f"{name}: using {fraction:.0%} of {total / 1024 ** 3:.1f} GB", flush=True)
                link.mkdir(parents=True, exist_ok=True)
                for number, mmap_path in enumerate(sorted(split.glob("**/*_mmap"))):
                    features = RaggedMmap(str(mmap_path))
                    count = max(1, int(len(features) * fraction))
                    indices = np.sort(rng.choice(len(features), size=count, replace=False))
                    RaggedMmap.from_generator(out_dir=str(link / f"subset{number}_mmap"),
                                              sample_generator=(features[int(i)] for i in indices),
                                              batch_size=1000, verbose=True)
            target.mkdir(parents=True, exist_ok=True)
            stamp.write_text("ok")
    return target_root


def stage_train(work, args):
    import yaml
    negatives = work / "negative_datasets"
    subsets = subset_negatives(work, args.negative_gb) if args.negative_gb > 0 else negatives
    features = work / "features"
    config = {
        "window_step_ms": 10,
        "train_dir": str(work / "trained_models" / f"hey_mirra_{MODEL_VERSION}"),
        "features": [
            *([{"features_dir": str(features / "real"), "sampling_weight": 1.0, "penalty_weight": 1.0, "truth": True,
                 "truncation_strategy": "truncate_start", "type": "mmap"}] if (features / "real").exists() else []),
            {"features_dir": str(features / "positive"), "sampling_weight": 2.0, "penalty_weight": 1.0, "truth": True,
             "truncation_strategy": "truncate_start", "type": "mmap"},
            {"features_dir": str(features / "adversarial"), "sampling_weight": 2.0, "penalty_weight": 1.0,
             "truth": False, "truncation_strategy": "truncate_start", "type": "mmap"},
            {"features_dir": str(subsets / "speech"), "sampling_weight": 10.0, "penalty_weight": 1.0,
             "truth": False, "truncation_strategy": "random", "type": "mmap"},
            {"features_dir": str(subsets / "dinner_party"), "sampling_weight": 10.0, "penalty_weight": 1.0,
             "truth": False, "truncation_strategy": "random", "type": "mmap"},
            {"features_dir": str(subsets / "no_speech"), "sampling_weight": 5.0, "penalty_weight": 1.0,
             "truth": False, "truncation_strategy": "random", "type": "mmap"},
            {"features_dir": str(negatives / "dinner_party_eval"), "sampling_weight": 0.0, "penalty_weight": 1.0,
             "truth": False, "truncation_strategy": "split", "type": "mmap"},
        ],
        # second phase with a lower learning rate, so the model settles instead of jumping between
        # "hears everything" and "hears nothing" from one validation to the next
        "training_steps": [args.steps, args.steps // 2],
        "positive_class_weight": [1, 1],
        "negative_class_weight": [20, 20],
        "learning_rates": [0.001, 0.0001],
        "batch_size": 128 if not args.smoke else 16,
        "time_mask_max_size": [0, 0],
        "time_mask_count": [0, 0],
        "freq_mask_max_size": [0, 0],
        "freq_mask_count": [0, 0],
        "eval_step_interval": 1000 if not args.smoke else 25,
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }
    config_path = work / "training_parameters.yaml"
    config_path.write_text(yaml.dump(config))

    python = args.train_python or sys.executable
    # TensorFlow does not find every CUDA library of its pip packages by itself (libcusolver.so.11 with
    # TensorFlow 2.21), so all of them are put on the library path of the training processes
    venv = Path(python).absolute().parent.parent if args.train_python else Path(sys.prefix)
    cuda_dirs = sorted(str(d) for d in venv.glob("lib/python3*/site-packages/nvidia/*/lib"))
    if cuda_dirs:
        TRAIN_ENV["LD_LIBRARY_PATH"] = ":".join(cuda_dirs + [os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    env = dict(TRAIN_ENV)
    if args.smoke or args.cpu or not gpu_usable(python):
        log("Training on the CPU")
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        log("Training on the GPU")
    command = [python, "-m", "microwakeword.model_train_eval",
               f"--training_config={config_path}", "--train", "1", "--restore_checkpoint", "1",
               "--test_tf_nonstreaming", "0", "--test_tflite_nonstreaming", "0",
               "--test_tflite_nonstreaming_quantized", "0", "--test_tflite_streaming", "0",
               "--test_tflite_streaming_quantized", "1", "--use_weights", "best_weights",
               "mixednet", "--pointwise_filters", "64,64,64,64", "--repeat_in_block", "1, 1, 1, 1",
               "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]", "--residual_connection", "0,0,0,0",
               "--first_conv_filters", "32", "--first_conv_kernel_size", "5", "--stride", "3"]
    with open(work / "training.log", "a", encoding="utf-8") as logfile:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   cwd=work)
        for line in process.stdout:
            print(line, end="", flush=True)
            logfile.write(line)
        if process.wait() != 0:
            raise SystemExit("Training failed, see training.log")


# --------------------------------------------------------------------------- export

def stage_export(work, out):
    trained = work / "trained_models" / f"hey_mirra_{MODEL_VERSION}"
    model = trained / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
    if not model.exists():
        raise SystemExit(f"No model at {model}")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(model, out / "hey_mirra.tflite")
    results = out / "results"
    results.mkdir(exist_ok=True)
    for file in trained.rglob("*.txt"):
        shutil.copy(file, results / file.name)
    for name in ("training.log", "training_parameters.yaml"):
        if (work / name).exists():
            shutil.copy(work / name, results / name)
    manifest = {
        "type": "micro", "wake_word": "Hey Mirra", "author": "MirraNET", "website": "https://mirranet.de",
        "model": "hey_mirra.tflite", "trained_languages": ["de", "en"], "version": 2,
        "micro": {"probability_cutoff": 0.9, "sliding_window_size": 5, "feature_step_size": 10,
                  "tensor_arena_size": 30000, "minimum_esphome_version": "2024.7.0"},
    }
    (out / "hey_mirra.json").write_text(json.dumps(manifest, indent=2))
    write_header(out / "hey_mirra.tflite", out / "hey_mirra_model.h")
    log(f"Model written to {out / 'hey_mirra.tflite'}")


def write_header(model, header):
    """C array of the model for the firmware (copy next to mirra-voice.ino)."""
    data = model.read_bytes()
    lines = [", ".join(f"0x{b:02x}" for b in data[i:i + 16]) for i in range(0, len(data), 16)]
    header.write_text(
        "// Wake word model \"Hey Mirra\" (microWakeWord), generated by mww_train.py\n"
        "#pragma once\n\n"
        "alignas(16) const unsigned char hey_mirra_tflite[] = {\n  " + ",\n  ".join(lines) + "\n};\n"
        f"const unsigned int hey_mirra_tflite_len = {len(data)};\n")


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="tiny offline run to test the pipeline")
    parser.add_argument("--cpu", action="store_true", help="train on the CPU even if a GPU is available")
    parser.add_argument("--train-python", help="Python of the TensorFlow environment used for training")
    parser.add_argument("--only", choices=["samples", "data", "features", "train", "export"])
    parser.add_argument("--positive-de", type=int, default=20000)
    parser.add_argument("--positive-en", type=int, default=6000)
    parser.add_argument("--adversarial-de", type=int, default=12000)
    parser.add_argument("--variants-de", type=int, default=8000)
    parser.add_argument("--variants-en", type=int, default=2000)
    parser.add_argument("--adversarial-en", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20000, help="steps of the first phase, the second has half")
    parser.add_argument("--negative-gb", type=float, default=3.0,
                        help="RAM budget for the negative training sets (0 = use all, needs a lot of RAM)")
    parser.add_argument("--audioset-parts", type=int, default=2, help="number of AudioSet files for background noise")
    args = parser.parse_args()
    if args.smoke:
        args.positive_de, args.positive_en, args.adversarial_de, args.adversarial_en = 60, 20, 40, 10
        args.variants_de, args.variants_en = 20, 10
        args.batch_size, args.steps = 10, 60
        args.negative_gb = 0.0005

    work = args.work.expanduser().resolve()
    out = args.out.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    random.seed(7)

    migrate(work)
    stages = [("samples", lambda: stage_samples(work, args)), ("data", lambda: stage_data(work, args)),
              ("recordings", lambda: stage_recordings(work, out) if recordings_changed(work, out) else None),
              ("features", lambda: stage_features(work, args)), ("train", lambda: stage_train(work, args)),
              ("export", lambda: stage_export(work, out))]
    for name, function in stages:
        if args.only and name != args.only:
            continue
        if done(work, name) and not args.only and name != "recordings":
            print(f"stage {name}: already done")
            continue
        log(f"Stage {name}")
        function()
        mark(work, name)
    log("Finished")


if __name__ == "__main__":
    main()
