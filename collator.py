from torch import randint, stack
from torch.nn.functional import pad
from torchaudio.transforms import Resample


class Collator:
    def __init__(self, *, source_sample_rate, target_sample_rate, segment_length):
        self.resample = Resample(source_sample_rate, target_sample_rate)
        self.segment_length = segment_length
        self.max_offset = target_sample_rate // 150

    def __call__(self, batch):
        resampled_batch = []
        shifted_batch = []

        for waveform, *_ in batch:
            waveform = self.resample(waveform)
            offset = int(randint(1, self.max_offset, ()).item())
            shifted_waveform = pad(waveform, (offset, 0))

            if waveform.size(-1) <= self.segment_length:
                waveform = pad(waveform, (0, self.segment_length - waveform.size(-1)))

                if shifted_waveform.size(-1) <= self.segment_length:
                    shifted_waveform = pad(
                        shifted_waveform,
                        (0, self.segment_length - shifted_waveform.size(-1)),
                    )
                else:
                    shifted_waveform = shifted_waveform[..., : self.segment_length]

            else:
                offset = randint(
                    0, waveform.size(-1) - self.segment_length + 1, ()
                ).item()
                waveform = waveform[..., offset : offset + self.segment_length]
                shifted_waveform = shifted_waveform[
                    ..., offset : offset + self.segment_length
                ]

            resampled_batch.append(waveform)
            shifted_batch.append(shifted_waveform)

        resampled_batch = stack(resampled_batch)
        shifted_batch = stack(shifted_batch)

        return resampled_batch, shifted_batch
