import random
import math
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


def jitter(x, jitter_size, dims=None, jitter_batches=1):
    if dims is None:
        dims = [len(x.shape)-1]

    sizes = [x.shape[d] - jitter_size[e] for e, d in enumerate(dims)]

    x = x.unsqueeze(0)
    rep = [1] * len(x.shape)
    rep[0] = jitter_batches
    x = x.repeat(rep)
    dims = [d+1 for d in dims]

    indices = [torch.arange(s).unsqueeze(0).repeat(jitter_batches, 1) for s in sizes]

    for b in range(jitter_batches):
        xs = x[b]
        for e, d in enumerate(dims):
            o = random.randint(jitter_size[d])
            xs = torch.index_select(xs, d, indices=o)


class Jitter(torch.nn.Module):
    def __init__(self, jitter_sizes, dims, jitter_batches=1):
        super().__init__()
        self.jitter_sizes = jitter_sizes
        self.dims = dims
        self.jitter_batches = jitter_batches
        self.verbose = 0

    def forward(self, x):
        if x.shape[0] == 1 and self.jitter_batches > 1:
            jitter_batches = self.jitter_batches
            x = x.repeat([self.jitter_batches] + [1]*(len(x.shape)-1))
        else:
            jitter_batches = x.shape[0]

        for e, d in enumerate(self.dims):
            if self.jitter_sizes[e] < 1:
                continue
            size = x.shape[d] - self.jitter_sizes[e]
            indices = torch.arange(size).unsqueeze(0).repeat(jitter_batches, 1)
            offset = torch.randint(self.jitter_sizes[e], size=[jitter_batches]).unsqueeze(1)
            if self.verbose >= 1:
                print("jitter offsets in dimension " + str(d) + ": " + str(offset))
            indices += offset

            for ud in range(1, len(x.shape)):
                if ud == d:
                    continue
                r = [1] * (len(indices.shape)+1)
                r[ud] = x.shape[ud]
                indices = indices.unsqueeze(ud).repeat(r)

            indices = indices.to(x.device)
            x = torch.gather(x, d, indices)

        return x


class JitterLoop(torch.nn.Module):
    def __init__(self, output_length, dim, jitter_size=None, jitter_batches=1):
        super().__init__()
        self.output_length = output_length
        self.dim = dim
        self.jitter_batches = jitter_batches
        self.jitter_size = jitter_size

    def forward(self, x):
        if x.shape[0] == 1 and self.jitter_batches > 1:
            jitter_batches = self.jitter_batches
            x = x.repeat([self.jitter_batches] + [1] * (len(x.shape) - 1))
        else:
            jitter_batches = x.shape[0]

        # repeat x as often as needed and return a section with the specified length
        repeats = 1 + math.ceil(self.output_length / x.shape[self.dim])
        repeat_list = [1] * len(x.shape)
        repeat_list[self.dim] = repeats
        rx = x.repeat(repeat_list)

        if self.jitter_size is None:
            jitter_size = x.shape[self.dim]
        else:
            jitter_size = self.jitter_size

        indices = torch.arange(self.output_length).unsqueeze(0).repeat(jitter_batches, 1)
        offset = torch.randint(jitter_size, size=[jitter_batches]).unsqueeze(1)

        indices += offset

        for ud in range(1, len(rx.shape)):
            if ud == self.dim:
                continue
            r = [1] * (len(indices.shape) + 1)
            r[ud] = rx.shape[ud]
            indices = indices.unsqueeze(ud).repeat(r)

        indices = indices.to(rx.device)
        new_x = torch.gather(rx, self.dim, indices)
        return new_x


def mask_height_section(x, size, channel=None, value=0.):
    for i in range(x.shape[0]):
        masking_size = random.randint(0, size-1)
        position_range = x.shape[2] - masking_size
        position = random.randint(0, position_range-1)
        if channel is None:
            x[i, :, position:position+masking_size, :] *= 0. + value
        else:
            x[i, channel, position:position + masking_size, :] *= 0. + value
    return x


def mask_width_section(x, size, channel=None, value=0.):
    for i in range(x.shape[0]):
        masking_size = random.randint(0, size-1)
        position_range = x.shape[3] - masking_size
        position = random.randint(0, position_range-1)
        if channel is None:
            x[i, :, :, position:position+masking_size] *= 0. + value
        else:
            x[i, channel, :, position:position + masking_size] *= 0. + value
    return x


def istft(stft_matrix, hop_length=None, win_length=None, window='hann',
          center=True, normalized=False, onesided=True, length=None):
    # https://keunwoochoi.wordpress.com/2019/03/14/inverse-stft-in-pytorch/
    """stft_matrix = (batch, freq, time, complex)

    All based on librosa
        - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
    What's missing?
        - normalize by sum of squared window --> do we need it here?
        Actually the result is ok by simply dividing y by 2.
    """
    assert normalized == False
    assert onesided == True
    assert window == "hann"
    assert center == True

    device = stft_matrix.device
    n_fft = 2 * (stft_matrix.shape[-3] - 1)

    batch = stft_matrix.shape[0]

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    istft_window = torch.hann_window(n_fft).to(device).view(1, -1)  # (batch, freq)

    n_frames = stft_matrix.shape[-2]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    y = torch.zeros(batch, expected_signal_len, device=device)
    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, :, i]
        iffted = torch.irfft(spec, signal_ndim=1, signal_sizes=(win_length,))

        ytmp = istft_window * iffted
        y[:, sample:(sample + n_fft)] += ytmp

    y = y[:, n_fft // 2:]

    if length is not None:
        if y.shape[1] > length:
            y = y[:, :length]
        elif y.shape[1] < length:
            y = torch.cat(y[:, :length], torch.zeros(y.shape[0], length - y.shape[1], device=y.device))

    coeff = n_fft / float(
        hop_length) / 2.0  # -> this might go wrong if curretnly asserted values (especially, `normalized`) changes.
    return y / coeff


def spectral_local_response_normalization(x, size=3, n_fft=512):
    x_stft = torch.stft(x, n_fft)
    amplitude = torch.sqrt(x_stft[:, :, :, 0]**2 + x_stft[:, :, :, 1]**2)
    x_stft_norm = torch.mean(amplitude, dim=1, keepdim=True)
    x_avg = F.avg_pool1d(x_stft_norm, kernel_size=size, stride=1, padding=size//2, count_include_pad=False)
    normalized_stft = x_stft / x_avg.unsqueeze(3)
    #normalized_stft = x_stft
    normalized_x = istft(normalized_stft)
    padding_length = (normalized_x.shape[1] - x.shape[1]) // 2
    normalized_x = normalized_x[:, :x.shape[1]]
    return normalized_x
