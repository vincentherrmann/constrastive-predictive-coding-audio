import random
import math
import torch
import torch.nn.functional as F
import numpy as np
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


class ActivationNormalization(torch.nn.Module):
    def __init__(self, means, variances):
        super().__init__()
        self.means = torch.nn.ParameterDict({key: torch.nn.Parameter(value.unsqueeze(0), requires_grad=False)
                                             for (key, value) in  means.items()})
        self.variances = torch.nn.ParameterDict({key: torch.nn.Parameter(value.unsqueeze(0), requires_grad=False)
                                                 for (key, value) in  variances.items()})

    def forward(self, activations):
        for key, value in self.means.items():
            activations[key] = (activations[key] - value) / self.variances[key]
        return activations


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
    def __init__(self, output_length, dim, jitter_size=None, jitter_batches=1, first_batch_offset=None):
        super().__init__()
        self.output_length = output_length
        self.dim = dim
        self.jitter_batches = jitter_batches
        self.jitter_size = jitter_size
        self.first_batch_offset = first_batch_offset

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
        if self.jitter_size <= 0:
            offset = torch.zeros(jitter_batches, dtype=torch.long).unsqueeze(1)
        else:
            offset = torch.randint(jitter_size, size=[jitter_batches]).unsqueeze(1)

        if self.first_batch_offset is not None:
            offset[0] = self.first_batch_offset

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


class Masking2D(torch.nn.Module):
    def __init__(self, size, axis='width', value=0., exclude_first_batch=False):
        super().__init__()
        self.axis = axis
        self.size = size
        self.value = value
        self.exclude_first_batch = exclude_first_batch

    def forward(self, x):
        if self.size < 1:
            return x

        dim = 3 if self.axis == 'width' else 2

        new_x = x.clone()

        try:
            if self.value.shape[0] != x.shape[1]:
                print("value has wrong shape, x has ", x.shape[1], "channels")
            value = self.value.to(x.device)
        except:
            value = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device) + self.value

        for batch in range(1 if self.exclude_first_batch else 0, x.shape[0]):
            #masking_size = random.randint(0, self.size-1)
            masking_size = self.size
            if masking_size == 0:
                continue
            position_range = x.shape[dim] - masking_size
            position = random.randint(0, position_range - 1)
            if self.axis == 'width':
                v = value.view(-1, 1, 1).repeat([1, x.shape[2], masking_size])
                new_x[batch, :, :, position:position+masking_size] = v
            elif self.axis == 'height':
                v = value.view(-1, 1, 1).repeat([1, masking_size, x.shape[3]])
                new_x[batch, :, position:position + masking_size, :] = v
            else:
                print("unknown axis:", self.axis)

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


def window_function(x, min_clamp=-1., max_clamp=1.):
    x = np.clip(x, min_clamp, max_clamp)
    return (2 - 2*np.cos(np.pi*(x+1))) * 0.25


def eq_bands(levels, freqs, sizes, fft_bands, sample_rate=16000):
    base_f = 0.5 * sample_rate / fft_bands
    scaling = 1.
    x = np.linspace(0., 1., num=fft_bands, dtype=np.float32)
    x_freq = np.linspace(base_f, sample_rate/2, num=fft_bands, dtype=np.float32)
    x_log = np.geomspace(x_freq[0], x_freq[-1], endpoint=True, num=fft_bands, dtype=np.float32)

    all_bands = 0.
    for i in range(len(levels)):
        band_x = (x - freqs[i] * scaling) / (sizes[i] * scaling)
        all_bands += levels[i] * window_function(band_x,
                                     min_clamp=0. if i == 0 else -1.,
                                     max_clamp=0. if i == len(levels)-1 else 1.)
    all_bands = np.interp(x_freq, x_log, all_bands).astype(np.float32)
    return 10**(all_bands / 20), x_freq


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


def activation_downsampling(activation_dict, target_length):
    activation_dict = activation_dict.copy()
    for key, value in activation_dict.items():
        if key == 'c_code' or key == 'prediction':
            continue
        dim = len(value.shape) - 1
        downsampling_factor = int(value.shape[dim] / target_length)
        if downsampling_factor > 1:
            unsqueeze = len(value.shape) == 2
            if unsqueeze:
                value = value.unsqueeze(1)
            activation_dict[key] = F.avg_pool1d(value, kernel_size=downsampling_factor)
            if unsqueeze:
                value.squeeze(1)

    return activation_dict


def convert_activation_dict_type(activation_dict, dtype=torch.float, select_batch=None):
    converted_activations = {}
    if select_batch is not None:
        for key, value in activation_dict.items():
            converted_activations[key] = value[select_batch].detach().cpu().contiguous().clone().type(dtype)
    else:
        for key, value in activation_dict.items():
            converted_activations[key] = value.detach().cpu().contiguous().clone().type(dtype)
    return converted_activations


def interpolate_position(x, pos, dim=None):
    if dim is None:
        dim = len(x.shape) - 1

    length = x.shape[dim]
    if length == 1:
        return x.squeeze(dim)

    pos = min(max(0., pos), 1.)
    idx = int(pos*length)
    idx2 = idx + 1 if idx < length-1 else 0
    interp = pos*length - idx

    a = torch.index_select(x, dim=dim, index=torch.LongTensor([idx])).squeeze(dim)
    b = torch.index_select(x, dim=dim, index=torch.LongTensor([idx2])).squeeze(dim)
    return (1 - interp) * a + interp * b


def select_activation_slice(activations, channel=0., channel_region=1.,
                            pitch=0., pitch_region=1.,
                            time=0., time_region=1.):

    # batch, channel, pitch, time  or
    # batch, channel, time

    num_channels = activations.shape[1]
    channel_region = max(1, int(channel_region * num_channels))
    channel_pos = int(channel * (num_channels - channel_region + 1))
    channel_start = max(0, channel_pos - channel_region + 1)
    channel_end = min(num_channels, channel_pos + channel_region)

    num_time = activations.shape[-1]
    time_region = max(1, int(time_region * num_time))
    time_pos = int(time * (num_time - time_region + 1))
    time_start = max(0, time_pos - time_region + 1)
    time_end = min(num_time, time_pos + time_region)

    if len(activations.shape) == 3:
        slice = activations[:, channel_start:channel_end, time_start:time_end]
    else:
        num_pitch = activations.shape[2]
        pitch_region = max(1, int(pitch_region * num_pitch))
        pitch_pos = int(pitch * (num_pitch - pitch_region + 1))
        pitch_start = max(0, pitch_pos - pitch_region + 1)
        pitch_end = min(num_pitch, pitch_pos + pitch_region)
        slice = activations[:, channel_start:channel_end, pitch_start:pitch_end, time_start:time_end]

    return slice


def flatten_activations(activation_dict, exclude_first_dimension=False):
    activations = []
    if exclude_first_dimension:
        for key, value in activation_dict.items():
            activations.append(value.view(value.shape[0], -1))
        activations = torch.cat(activations, dim=1)
    else:
        for key, value in activation_dict.items():
            activations.append(value.view(-1))
        activations = torch.cat(activations)
    return activations

