import torch
import torchaudio
from torchaudio.datasets import LIBRITTS

from model import Model

sample_rate = 16_000
dataset = LIBRITTS("./data/", "test-clean")

with torch.inference_mode():
    model = Model(channel_count=64)
    state_dict = torch.load("checkpoints/208000/model.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    audio = dataset[42]

    audio = torchaudio.functional.resample(
        audio[0][0], audio[1], sample_rate
    ).unsqueeze(0)

    latent = model.encoder(audio)

    with open("out/lat.txt", "w") as file:
        for xs in latent[0].T:
            print("".join(str(int(x + 1)) for x in xs), file=file)

    water = []
    input("Modify latents")

    with open("out/lat.txt") as file:
        for line in file.readlines():
            water.append([int(x) - 1 for x in line.strip()])

    water = torch.tensor(water, dtype=torch.float32).T

    rec = model.decoder(latent)
    torchaudio.save("out/rec.wav", rec.squeeze(0), sample_rate)

    audio = audio[:, : rec.size(-1)]
    torchaudio.save("out/audio.wav", audio, sample_rate)

    wat = model.decoder(water)
    torchaudio.save("out/wat.wav", wat.squeeze(0), sample_rate)

    new = model.encoder(wat)

    with open("out/new.txt", "w") as file:
        for xs in new[0].T:
            print("".join(str(int(x + 1)) for x in xs), file=file)
