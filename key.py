from math import comb
from random import Random


class Key:
    def __init__(
        self,
        *,
        channel_count: int,
        kernel_size: int,
        confidence_factor: float,
        seed: int,
        source_count: int,
        target_count: int,
    ):
        self.channel_count = channel_count
        self.confidence_factor = confidence_factor
        self.kernel_size = kernel_size

        points = [
            (row, column)
            for row in range(kernel_size)
            for column in range(channel_count)
        ]
        selected_points = Random(seed).sample(points, source_count + target_count)
        self.sources = selected_points[:source_count]
        self.targets = selected_points[source_count:]

    def check(self, latents: list[list[int]], apply: bool = False) -> float:
        count = 0
        shrunk_length = len(latents) + 1 - self.kernel_size

        for i in range(shrunk_length):
            pattern = latents[i : i + self.kernel_size]
            value = 0

            for row, column in self.sources:
                value += pattern[row][column]

            value = (value % 3) - 1
            match = True

            for row, column in self.targets:
                if pattern[row][column] != value:
                    match = False

                    if apply:
                        pattern[row][column] = value

                value = (value + 2) % 3 - 1

            if match:
                count += 1

        p = self.confidence_factor / 3 ** len(self.targets)

        return 1 - sum(
            comb(shrunk_length, k) * p**k * (1 - p) ** (shrunk_length - k)
            for k in range(count, shrunk_length + 1)
        )
