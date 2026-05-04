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
        channels = random.sample(range(channel_count), 5)
        self.gate_channel = channels[0]
        self.minion_channels = []

        for _ in range(3):
            batch = channels[1:]
            random.shuffle(batch)
            self.minion_channels.append(batch)

    def check(self, latents: list[list[int]], apply: bool = False) -> float:
        count = 0
        shrunk_length = len(latents) - 2

        for i in range(shrunk_length):
            gate = latents[i][self.gate_channel]
            minion_channels = self.minion_channels[gate]

            if (
                latents[i][minion_channels[0]] > latents[i][minion_channels[1]]
                and latents[i][minion_channels[2]] > latents[i][minion_channels[3]]
            ):
                count += 1

            if apply:
                for j in [minion_channels[0], minion_channels[2]]:
                    latents[i][j] = max(
                        latents[i][j], latents[i + 1][j], latents[i + 2][j]
                    )

                for j in [minion_channels[1], minion_channels[3]]:
                    latents[i][j] = min(
                        latents[i][j], latents[i + 1][j], latents[i + 2][j]
                    )

        p = self.confidence_factor / 9

        return 1 - sum(
            comb(shrunk_length, k) * p**k * (1 - p) ** (shrunk_length - k)
            for k in range(count, shrunk_length + 1)
        )
