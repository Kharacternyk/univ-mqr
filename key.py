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

        random = Random(seed)
        points = [
            (row, column)
            for row in range(kernel_size)
            for column in range(channel_count)
        ]

        selected_points = random.sample(points, source_count + target_count)

        self.sources = selected_points[:source_count]
        self.targets = selected_points[source_count:]

        self.coefficients = [
            random.choices(range(3), k=source_count + 1) for _ in range(target_count)
        ]

    def check(self, latents: list[list[int]], apply: bool = False) -> float:
        count = 0
        shrunk_length = len(latents) + 1 - self.kernel_size

        for i in range(shrunk_length):
            pattern = latents[i : i + self.kernel_size]
            values = [pattern[row][column] for row, column in self.sources]

            match = True

            for (row, column), coefficients in zip(self.targets, self.coefficients):
                result = coefficients[-1]

                for value, coefficient in zip(values, coefficients[:-1]):
                    result += value * coefficient

                result = (result % 3) - 1

                if pattern[row][column] != result:
                    match = False

                    if apply:
                        pattern[row][column] = result

            if match:
                count += 1

        p = self.confidence_factor / 3 ** len(self.targets)

        return 1 - sum(
            comb(shrunk_length, k) * p**k * (1 - p) ** (shrunk_length - k)
            for k in range(count, shrunk_length + 1)
        )
