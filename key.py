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
        self.random = Random(seed)
        self.channels = self.random.sample(range(channel_count), 3)

    def check(self, latents: list[list[int]], apply: bool = False) -> float:
        count = 0

        for latent in latents:
            for i in self.channels:
                if latent[i] == 0:
                    count += 1
                elif apply:
                    latent[i] = 0

        p = 0.55
        n = len(latents) * 3
        mu = n * p
        sigma = sqrt(n * p * (1 - p))
        t = (count - mu) / sigma

        return (1 + erf(t / sqrt(2))) / 2
