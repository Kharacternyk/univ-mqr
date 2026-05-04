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

        random = Random(seed)
        self.x, self.y = random.sample(range(channel_count), 2)

    def check(self, latents: list[list[int]], apply: bool = False) -> float:
        count = 0
        shrunk_length = len(latents) - 2

        for i in range(shrunk_length):
            if latents[i][self.x] > latents[i][self.y]:
                count += 1

            if apply:
                latents[i][self.x] = max(
                    latents[i][self.x], latents[i + 1][self.x], latents[i + 2][self.x]
                )
                latents[i][self.y] = min(
                    latents[i][self.y], latents[i + 1][self.y], latents[i + 2][self.y]
                )

        p = self.confidence_factor / 3

        return 1 - sum(
            comb(shrunk_length, k) * p**k * (1 - p) ** (shrunk_length - k)
            for k in range(count, shrunk_length + 1)
        )
