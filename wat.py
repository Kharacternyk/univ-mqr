from pathlib import Path
from random import Random
from subprocess import run

import torch
import torchaudio
from torchaudio.datasets import LIBRITTS
from tqdm import tqdm

from key import Key
from model import Model

key = Key(
    channel_count=64,
    kernel_size=5,
    confidence_factor=1.5,
    seed=0,
    source_count=2,
    target_count=3,
)

sample_rate = 16_000
dataset = LIBRITTS("./data/", "test-clean")

with torch.inference_mode():
    model = Model(channel_count=64).cuda()
    state_dict = torch.load("checkpoints/208000/model.pt")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    audios = [dataset[i] for i in Random(0).sample(range(len(dataset)), 5)]

    for audio, original_sample_rate, *_, sample_id in tqdm(audios):
        sample_dir = Path("wat") / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        audio = torchaudio.functional.resample(
            audio.cuda(), original_sample_rate, sample_rate
        )

        latent_tensor = model.encoder(audio)
        latent_list = [[int(x) for x in xs] for xs in latent_tensor[0].T]

        rec = model.decoder(latent_tensor)
        torchaudio.save(sample_dir / "rec.wav", rec.squeeze(0).cpu(), sample_rate)

        audio = audio[:, : rec.size(-1)]
        torchaudio.save(sample_dir / "orig.wav", audio.cpu(), sample_rate)

        orig_score = key.check(latent_list, apply=True)
        wat_tensor = torch.tensor(latent_list, dtype=torch.float32, device="cuda").T

        wat_audio = model.decoder(wat_tensor)
        torchaudio.save(sample_dir / "wat.wav", wat_audio.squeeze(0).cpu(), sample_rate)

        wat_tensor = model.encoder(wat_audio)
        wat_list = [[int(x) for x in xs] for xs in wat_tensor[0].T]

        new_score = key.check(wat_list)

        shifted_audio = torch.nn.functional.pad(wat_audio[0], (8, 0))
        shifted_tensor = model.encoder(shifted_audio)
        shifted_list = [[int(x) for x in xs] for xs in shifted_tensor[0].T]
        shift_score = key.check(shifted_list)

        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i",
                sample_dir / "wat.wav",
                "-c:a",
                "libopus",
                "-b:a",
                "16k",
                sample_dir / "wat-opus.ogg",
            ],
            check=True,
            capture_output=True,
        )

        opus, original_sample_rate = torchaudio.load(sample_dir / "wat-opus.ogg")
        opus = torchaudio.functional.resample(
            opus[0].cuda(), original_sample_rate, sample_rate
        ).unsqueeze(0)

        opus_tensor = model.encoder(opus)
        opus_list = [[int(x) for x in xs] for xs in opus_tensor[0].T]

        opus_score = key.check(opus_list)

        print(f"Orig: {orig_score:.2f}")
        print(f"New: {new_score:.2f}")
        print(f"Opus: {opus_score:.2f}")
        print(f"Shift: {shift_score:.2f}")
