from pathlib import Path
from random import Random

import torchaudio
from torch.nn.functional import pad
from torchaudio.datasets import LIBRITTS
from tqdm import tqdm

from opus import convert_to_opus
from watermark import Watermark

dataset = LIBRITTS("./data/", "test-clean")
indices = Random(1).sample(range(len(dataset)), 150)
subset = [dataset[i] for i in indices]

watermark = Watermark(
    checkpoint="checkpoints/208000/model.pt",
    channel_count=64,
    sample_rate=16_000,
    shift=16,
)

audio_filename = "audio.wav"
applied_filename = "applied.wav"
reconstructed_filename = "reconstructed.wav"

for audio, sample_rate, *_, sample_id in tqdm(subset):
    sample_dir = Path("test") / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    seed = hash(sample_id)

    original_score, audio, applied, reconstructed = watermark.check(
        audio=audio, seed=seed, sample_rate=sample_rate, apply=True, reconstruct=True
    )

    assert applied is not None
    assert reconstructed is not None

    torchaudio.save(sample_dir / audio_filename, audio, watermark.sample_rate)
    torchaudio.save(sample_dir / applied_filename, applied, watermark.sample_rate)
    torchaudio.save(
        sample_dir / reconstructed_filename, reconstructed, watermark.sample_rate
    )

    applied_score = watermark.check(audio=applied, seed=seed)[0]

    shifted_audio = pad(applied, (Random(seed).randint(0, 320), 0))
    shifted_score = watermark.check(audio=shifted_audio, seed=seed)[0]

    opus_scores = []

    for bitrate in ["8k", "10k"]:
        opus_filename = sample_dir / f"applied-{bitrate}.ogg"
        convert_to_opus(
            source_filename=sample_dir / applied_filename,
            target_filename=opus_filename,
            bitrate=bitrate,
        )

        opus, sample_rate = torchaudio.load(opus_filename)
        opus_scores.append(
            watermark.check(audio=opus, seed=seed, sample_rate=sample_rate)[0]
        )

    print(
        ",".join(
            [
                sample_id,
                f"{original_score:.2f}",
                f"{applied_score:.2f}",
                f"{shifted_score:.2f}",
                f"{opus_scores[0]:.2f}",
                f"{opus_scores[1]:.2f}",
            ]
        ),
        flush=True,
    )

    for bitrate in ["6k", "8k", "12k", "24k"]:
        opus_filename = sample_dir / f"audio-{bitrate}.ogg"
        convert_to_opus(
            source_filename=sample_dir / audio_filename,
            target_filename=opus_filename,
            bitrate=bitrate,
        )
