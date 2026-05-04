from torch import Tensor, float32, inference_mode, load, stack, tensor
from torch.nn.functional import pad
from torchaudio.functional import resample

from key import Key
from model import Model


class Watermark:
    @inference_mode()
    def __init__(self, *, checkpoint: str, key: Key, sample_rate: int, shift: int):
        state_dict = load(checkpoint, weights_only=True)
        model = Model(channel_count=key.channel_count).cuda()
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        # model.encoder.compile(fullgraph=True, dynamic=True, mode="reduce-overhead")
        # model.decoder.compile(fullgraph=True, dynamic=True, mode="reduce-overhead")

        self.model = model
        self.key = key
        self.sample_rate = sample_rate
        self.shift = shift

    @inference_mode()
    def check(
        self,
        *,
        audio: Tensor,
        sample_rate: int | None = None,
        apply: bool = False,
        reconstruct: bool = False,
    ) -> tuple[float, None | Tensor, None | Tensor, None | Tensor]:
        audio = audio.cuda()

        if sample_rate and sample_rate != self.sample_rate:
            audio = resample(audio, sample_rate, self.sample_rate)

        half_window = self.sample_rate // 100
        batch = stack(
            [
                pad(audio, (x, half_window - x))
                for x in range(0, half_window, self.shift)
            ]
        )

        latents_batch = [
            [[int(x) for x in xs] for xs in xss.T] for xss in self.model.encoder(batch)
        ]
        original_latents = tensor(latents_batch[0], dtype=float32, device="cuda").T

        high_score = 0

        for i, latents in enumerate(latents_batch):
            score = self.key.check(latents, apply=(i == 0 and apply))
            high_score = max(high_score, score)

        applied = None

        if apply:
            applied_latents = tensor(latents_batch[0], dtype=float32, device="cuda").T
            applied = self.model.decoder(applied_latents).cpu().squeeze(0)

        reconstructed = None
        trimmed = None

        if reconstruct:
            reconstructed = self.model.decoder(original_latents).cpu().squeeze(0)
            trimmed = audio[: len(reconstructed)].cpu()

        return high_score, trimmed, applied, reconstructed
