from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from random import Random

import torchaudio
from torch.nn.functional import pad
from torchaudio.datasets import LIBRITTS
from tqdm import tqdm

from opus import convert_to_opus
from watermark import Watermark

dataset = LIBRITTS("./data/", "test-clean")
indices = list(range(len(dataset)))
Random(-1).shuffle(indices)

watermark = Watermark(
    checkpoint="checkpoints/208000/model.pt",
    channel_count=64,
    sample_rate=16_000,
    shift=16,
)

audio_filename = "audio.wav"
applied_filename = "applied.wav"
reconstructed_filename = "reconstructed.wav"
bitrates = ["6k", "8k", "10k", "16k", "24k"]

for i in tqdm(indices):
    audio, sample_rate, *_, sample_id = dataset[i]
    sample_dir = Path("test") / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    original_score, latent_length, audio, applied, reconstructed = watermark.check(
        audio=audio,
        seed=i,
        sample_rate=sample_rate,
        apply=True,
        reconstruct=True,
    )

    assert applied is not None
    assert reconstructed is not None

    torchaudio.save(sample_dir / audio_filename, audio, watermark.sample_rate)
    torchaudio.save(sample_dir / applied_filename, applied, watermark.sample_rate)
    torchaudio.save(
        sample_dir / reconstructed_filename, reconstructed, watermark.sample_rate
    )

    with ThreadPoolExecutor() as executor:
        opus_applied_filenames = {}

        for bitrate in bitrates:
            opus_applied_filename = sample_dir / f"applied-{bitrate}.ogg"
            future = executor.submit(
                convert_to_opus,
                source_filename=sample_dir / applied_filename,
                target_filename=opus_applied_filename,
                bitrate=bitrate,
            )
            opus_applied_filenames[future] = opus_applied_filename, bitrate

            executor.submit(
                convert_to_opus,
                source_filename=sample_dir / audio_filename,
                target_filename=sample_dir / f"audio-{bitrate}.ogg",
                bitrate=bitrate,
            )

        applied_score = watermark.check(audio=applied, seed=i)[0]

        shifted_audio = pad(applied, (Random(i).randint(0, 320), 0))
        shifted_score = watermark.check(audio=shifted_audio, seed=i)[0]

        opus_scores = {}

        for future in as_completed(opus_applied_filenames):
            opus_filename, bitrate = opus_applied_filenames[future]
            opus, sample_rate = torchaudio.load(opus_filename)
            opus_scores[bitrate] = watermark.check(
                audio=opus, seed=i, sample_rate=sample_rate
            )[0]

        print(
            ",".join(
                [
                    sample_id,
                    f"{latent_length}",
                    f"{original_score[0]:.4f}",
                    f"{applied_score[0]:.4f}",
                    f"{shifted_score[0]:.4f}",
                    *(f"{opus_scores[bitrate][0]:.4f}" for bitrate in bitrates),
                    f"{original_score[1]}",
                    f"{applied_score[1]}",
                    f"{shifted_score[1]}",
                    *(f"{opus_scores[bitrate][1]}" for bitrate in bitrates),
                ]
            ),
            flush=True,
        )
