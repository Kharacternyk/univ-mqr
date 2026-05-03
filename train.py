from itertools import count
from pathlib import Path

from encodec.msstftd import DiscriminatorSTFT
from torch import (
    backends,
    inference_mode,
    load,
    nn,
    save,
    set_float32_matmul_precision,
    stack,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRITTS
from tqdm import tqdm

import wandb
from collator import Collator
from losses import (
    ReconstructionLoss,
    get_d_loss,
    get_feat_loss,
    get_g_loss,
    get_i_loss,
    get_noisy_i_loss,
)
from model import Model
from period_discriminator import PeriodDiscriminator
from scale_discriminator import ScaleDiscriminator

backends.cudnn.benchmark = True
set_float32_matmul_precision("medium")

checkpoints = Path("checkpoints")

sample_rate = 16_000

resume_index = 0

lr = 1e-4

model = Model(channel_count=64).cuda()

if resume_index:
    model.load_state_dict(load(checkpoints / f"{resume_index:06}" / "model.pt"))

model.encoder.compile(fullgraph=True, dynamic=False, mode="reduce-overhead")
model.decoder.compile(fullgraph=True, dynamic=False, mode="reduce-overhead")
model.compile(fullgraph=True, dynamic=False, mode="reduce-overhead")

optimizer = AdamW(model.parameters(), lr, (0.8, 0.9), fused=True)

if resume_index:
    optimizer.load_state_dict(load(checkpoints / f"{resume_index:06}" / "optimizer.pt"))

    for p in optimizer.param_groups:
        p["lr"] = lr

get_rec_loss = ReconstructionLoss(sample_rate=sample_rate).cuda()

collator = Collator(
    source_sample_rate=24_000,
    target_sample_rate=sample_rate,
    segment_length=7_950,
)

dataloader = DataLoader(
    LIBRITTS("./data/", download=True),
    batch_size=32,
    shuffle=True,
    num_workers=3,
    pin_memory=True,
    collate_fn=collator,
    drop_last=True,
)

val_dataloader = DataLoader(
    LIBRITTS("./data/", "test-clean", download=True),
    batch_size=32,
    shuffle=False,
    num_workers=3,
    pin_memory=True,
    collate_fn=collator,
    drop_last=True,
)

discriminators = nn.ModuleList(
    [
        PeriodDiscriminator(period=2),
        PeriodDiscriminator(period=3),
        PeriodDiscriminator(period=5),
        PeriodDiscriminator(period=7),
        PeriodDiscriminator(period=11),
        ScaleDiscriminator(sample_rate=sample_rate, downsample_rate=2),
        ScaleDiscriminator(sample_rate=sample_rate, downsample_rate=4),
        DiscriminatorSTFT(32, n_fft=2048, win_length=2048, hop_length=512),
        DiscriminatorSTFT(32, n_fft=1024, win_length=1024, hop_length=256),
        DiscriminatorSTFT(32, n_fft=512, win_length=512, hop_length=128),
        DiscriminatorSTFT(32, n_fft=256, win_length=256, hop_length=64),
        DiscriminatorSTFT(32, n_fft=128, win_length=128, hop_length=32),
    ]
).cuda()

if resume_index:
    discriminators.load_state_dict(
        load(checkpoints / f"{resume_index:06}" / "discriminators.pt")
    )

d_optimizer = AdamW(discriminators.parameters(), lr, (0.8, 0.9), fused=True)

if resume_index:
    d_optimizer.load_state_dict(
        load(checkpoints / f"{resume_index:06}" / "d_optimizer.pt")
    )

    for p in d_optimizer.param_groups:
        p["lr"] = lr


with wandb.init("kharacternyk-team", project="mqr") as logger:
    for epoch in count():
        for index, (batch, shifted_batch) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            d_optimizer.zero_grad()

            batch = batch.cuda(non_blocking=True)
            shifted_batch = shifted_batch.cuda(non_blocking=True)

            output, codes = model(batch)
            output = output.clone()
            codes = codes.clone()

            for p in discriminators.parameters():
                p.requires_grad_(True)

            d_losses = []

            for discriminator in discriminators:
                d_losses.append(get_d_loss(batch, output, discriminator))

            stack(d_losses).mean().backward()

            d_optimizer.step()

            for p in discriminators.parameters():
                p.requires_grad_(False)

            g_losses = []
            feat_losses = []

            for discriminator in discriminators:
                g_losses.append(get_g_loss(output, discriminator))
                feat_losses.append(get_feat_loss(batch, output, discriminator))

            adv_loss = stack(g_losses + feat_losses).mean()
            rec_loss = get_rec_loss(batch, output)
            noisy_i_loss = get_noisy_i_loss(codes, model)
            shift_i_loss = get_i_loss(shifted_batch, codes, model)

            total_loss = adv_loss + rec_loss + noisy_i_loss + shift_i_loss
            total_loss.backward()

            optimizer.step()

            global_index = resume_index + epoch * len(dataloader) + index + 1

            if not global_index % 50:
                metrics = dict(
                    total_loss=total_loss,
                    rec_loss=rec_loss,
                    adv_loss=adv_loss,
                    noisy_i_loss=noisy_i_loss,
                    shift_i_loss=shift_i_loss,
                )

                for i in range(len(discriminators)):
                    metrics.update(
                        {
                            f"d_loss/{i}": d_losses[i],
                            f"g_loss/{i}": g_losses[i],
                            f"feat_loss/{i}": feat_losses[i],
                        }
                    )

                logger.log(metrics, step=global_index)

            if not global_index % 2000:
                checkpoint = checkpoints / f"{global_index:06}"
                checkpoint.mkdir(parents=True, exist_ok=True)

                save(model.state_dict(), checkpoint / "model.pt")
                save(discriminators.state_dict(), checkpoint / "discriminators.pt")
                save(optimizer.state_dict(), checkpoint / "optimizer.pt")
                save(d_optimizer.state_dict(), checkpoint / "d_optimizer.pt")

                model.eval()
                discriminators.eval()

                rec_losses = []
                adv_losses = []
                noisy_i_losses = []
                shift_i_losses = []

                with inference_mode():
                    for batch, shifted_batch in tqdm(val_dataloader):
                        g_losses = []
                        feat_losses = []
                        batch = batch.cuda(non_blocking=True)
                        shifted_batch = shifted_batch.cuda(non_blocking=True)

                        output, codes = model(batch)
                        output = output.clone()
                        codes = codes.clone()

                        for discriminator in discriminators:
                            g_losses.append(get_g_loss(output, discriminator))
                            feat_losses.append(
                                get_feat_loss(batch, output, discriminator)
                            )

                        rec_losses.append(get_rec_loss(batch, output))
                        adv_losses.append(stack(g_losses + feat_losses).mean())
                        noisy_i_losses.append(get_noisy_i_loss(codes, model))
                        shift_i_losses.append(get_i_loss(shifted_batch, codes, model))

                    rec_loss = stack(rec_losses).mean()
                    adv_loss = stack(adv_losses).mean()
                    noisy_i_loss = stack(noisy_i_losses).mean()
                    shift_i_loss = stack(shift_i_losses).mean()
                    total_loss = rec_loss + adv_loss + noisy_i_loss + shift_i_loss

                    logger.log(
                        {
                            "val/rec_loss": rec_loss,
                            "val/adv_loss": adv_loss,
                            "val/noisy_i_loss": noisy_i_loss,
                            "val/shift_i_loss": shift_i_loss,
                            "val/total_loss": total_loss,
                        },
                        step=global_index,
                    )

                model.train()
                discriminators.train()
