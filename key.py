from math import erf, sqrt
from random import Random


class Key:
    def __init__(
        self,
        *,
        channel_count: int,
        seed: int,
    ):
        self.channel_count = channel_count
        self.channels = Random(seed).sample(range(channel_count), 3)

    def check(self, latents: list[list[int]], apply: bool = False) -> tuple[float, int]:
        count = 0

        for latent in latents:
            for i in self.channels:
                if latent[i] == 0:
                    count += 1
                elif apply:
                    latent[i] = 0

        p = 0.57
        n = len(latents) * len(self.channels)
        mu = n * p
        sigma = sqrt(n * p * (1 - p))
        t = (count - mu) / sigma

        return (1 + erf(t / sqrt(2))) / 2, count
