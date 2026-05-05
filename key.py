from math import comb
from random import Random


class Key:
    def __init__(
        self,
        *,
        channel_count: int,
        confidence_factor: float,
        seed: int,
    ):
        self.channel_count = channel_count
        self.confidence_factor = confidence_factor

        self.random = Random(seed)
        self.channels = self.random.sample(range(channel_count), 2)

    def check(self, latents: list[list[int]], apply: bool = False) -> float:
        count = 0
        shrunk_length = len(latents) - 2

        for i in range(shrunk_length):
            match = 0

            for j in range(2):
                if latents[i][self.channels[j]]:
                    match += 1
                elif apply:
                    latents[i][self.channels[j]] = self.random.choice([-1, 1])

            if match == 2:
                count += 1

        p = self.confidence_factor * 4 / 9

        return 1 - sum(
            comb(shrunk_length, k) * p**k * (1 - p) ** (shrunk_length - k)
            for k in range(count, shrunk_length + 1)
        )
