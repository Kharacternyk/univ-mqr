from pathlib import Path
from random import Random
from subprocess import run

import torchaudio
from torch.nn.functional import pad
from torchaudio.datasets import LIBRITTS
from tqdm import tqdm

from key import Key
from watermark import Watermark

key = Key(
    channel_count=64,
    kernel_size=7,
    confidence_factor=1.8,
    seed=0,
    source_count=2,
    target_count=3,
)
watermark = Watermark(
    checkpoint="checkpoints/208000/model.pt",
    key=key,
    sample_rate=16_000,
    shift=6,
)

dataset = LIBRITTS("./data/", "test-clean")
subset = [dataset[i] for i in Random(0).sample(range(len(dataset)), 5)]

for audio, sample_rate, *_, sample_id in tqdm(subset):
    sample_dir = Path("test") / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    original_score, applied, reconstructed = watermark.check(
        audio=audio, sample_rate=sample_rate, apply=True, reconstruct=True
    )

    assert applied is not None
    assert reconstructed is not None

    print(f"{original_score=:.2f}")

    audio = audio[:, : reconstructed.size(-1)]

    torchaudio.save(sample_dir / "audio.wav", audio, watermark.sample_rate)
    torchaudio.save(sample_dir / "applied.wav", applied, watermark.sample_rate)
    torchaudio.save(
        sample_dir / "reconstructed.wav", reconstructed, watermark.sample_rate
    )

    applied_score = watermark.check(audio=applied)[0]
    print(f"{applied_score=:.2f}")

    shifted_audio = pad(applied, (123, 0))
    shifted_score = watermark.check(audio=shifted_audio)[0]
    print(f"{shifted_score=:.2f}")

    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            sample_dir / "applied.wav",
            "-c:a",
            "libopus",
            "-b:a",
            "16k",
            sample_dir / "applied.ogg",
        ],
        check=True,
        capture_output=True,
    )

    opus, sample_rate = torchaudio.load(sample_dir / "applied.ogg")
    opus_score = watermark.check(audio=opus, sample_rate=sample_rate)[0]

    print(f"{opus_score=:.2f}")
