import math
import torch
import torch.utils.data
import numpy as np
import librosa as lr
import bisect
import torchaudio
from pathlib import Path


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self,
                 location,
                 item_length,
                 unique_length=None,
                 sampling_rate=16000,
                 mono=True,
                 dtype=torch.FloatTensor):
        super().__init__()
        self.location = Path(location)
        self.sampling_rate = sampling_rate
        self.mono = mono
        self._item_length = item_length
        self._unique_length = item_length if unique_length is None else unique_length
        self._length = 0
        self.start_samples = [0]
        self.dtype = dtype
        self.dummy_load = False

        self.files = list_all_audio_files(self.location, allowed_types=['.wav'])
        self.calculate_length()

    @property
    def item_length(self):
        return self._item_length

    @item_length.setter
    def item_length(self, value):
        self._item_length = value
        self.calculate_length()

    @property
    def unique_length(self):
        return self._unique_length

    @unique_length.setter
    def unique_length(self, value):
        self._unique_length = value
        self.calculate_length()

    def load_file(self, file, frames=-1, start=0):
        if frames == -1:
            data, _ = torchaudio.load(file,
                                      normalization=True)
            return data.squeeze().type(self.dtype)
        if frames == 0:
            print("Error: zero frames requested")
        data, _ = torchaudio.load(file,
                                  normalization=True,
                                  num_frames=frames,
                                  offset=start)
        return data.squeeze().type(self.dtype)

    def calculate_length(self):
        """
        Calculate the number of items in this data sets.
        Additionally the start positions of each file are calculate in this method.
        """
        start_samples = [0]
        for idx in range(len(self.files)):
            file_data = self.load_file(str(self.files[idx]))
            start_samples.append(start_samples[-1] + file_data.shape[0])
        available_length = start_samples[-1] - (self.item_length - self.unique_length)
        self._length = math.floor(available_length / self.unique_length)
        self.start_samples = start_samples

    def load_sample(self, file_index, position_in_file, item_length):
        file_length = self.start_samples[file_index + 1] - self.start_samples[file_index]
        remaining_length = position_in_file + item_length + 1 - file_length
        if remaining_length < 0:
            sample = self.load_file(str(self.files[file_index]),
                                    frames=item_length + 1,
                                    start=position_in_file)
        else:
            this_sample = self.load_file(str(self.files[file_index]),
                                         frames=item_length - remaining_length,
                                         start=position_in_file)
            next_sample = self.load_sample(file_index + 1,
                                           position_in_file=0,
                                           item_length=remaining_length)
            sample = torch.cat((this_sample, next_sample))
        return sample

    def get_position(self, idx):
        """
        Calculate the file and the position in the file from the global dataset index
        """
        sample_index = idx * self.unique_length

        file_index = bisect.bisect_left(self.start_samples, sample_index) - 1
        if file_index < 0:
            file_index = 0
        if file_index + 1 >= len(self.start_samples):
            print("error: sample index " + str(sample_index) + " is to high. Results in file_index " + str(file_index))
        position_in_file = sample_index - self.start_samples[file_index]
        return file_index, position_in_file

    def __getitem__(self, idx):
        if self.dummy_load:
            sample = np.random.randn(self._item_length)
        else:
            file_index, position_in_file = self.get_position(idx)
            sample = self.load_sample(file_index, position_in_file, self._item_length)

        example = sample[:self._item_length]
        return example

    def get_segment(self, position, file_index, duration=None):
        """
        Convenience function to get a segment from a file
        :param position: position in the file in seconds
        :param file_index: index of the file
        :param duration: the duration of the segment in seconds (plus the receptive field). If 'None', then only one receptive field is returned.
        :return: the specified segment (without labels)
        """
        position_in_file = (position // self.sampling_rate) - self.start_samples[file_index]
        if duration is None:
            item_length = self._item_length
        else:
            item_length = int(duration * self.sampling_rate)
        segment = self.load_sample(file_index, position_in_file, item_length)
        return segment

    def __len__(self):
        return self._length


class AudioTestingDataset(AudioDataset):
    def __init__(self,
                 location,
                 item_length,
                 unique_length=None,
                 sampling_rate=16000,
                 mono=True,
                 dtype=torch.FloatTensor,
                 max_file_count=None):

        super().__init__(location=location,
                         item_length=item_length,
                         unique_length=unique_length,
                         sampling_rate=sampling_rate,
                         mono=mono,
                         dtype=dtype)

        if max_file_count is None:
            self.max_file_count = len(self.files)
        else:
            self.max_file_count = max_file_count

    def calculate_length(self):
        """
        Calculate the number of items in this data sets.
        Additionally the start positions of each file are calculate in this method.
        Omit last sample in each file to make sure that there are no file-crossing samples.
        """
        start_samples = [0]
        for idx in range(self.max_file_count):
            file_data = self.load_file(str(self.files[idx]))
            start_samples.append(start_samples[-1] + file_data.shape[0] - self.item_length)
        available_length = start_samples[-1] - (self.item_length - self.unique_length)
        self._length = math.floor(available_length / self.unique_length)
        self.start_samples = start_samples

    def __getitem__(self, idx):
        if self.dummy_load:
            sample = np.random.randn(self._item_length)
        else:
            file_index, position_in_file = self.get_position(idx)
            sample = self.load_sample(file_index, position_in_file, self._item_length)

        example = sample[:self._item_length], torch.LongTensor([file_index]).squeeze()
        return example


def list_all_audio_files(location, allowed_types=[".mp3", ".wav", ".aif", "aiff", ".flac"]):
    types = allowed_types
    audio_files = []
    for type in types:
        audio_files.extend(sorted(location.glob('**/*' + type)))
    if len(audio_files) == 0:
        print("found no audio files in " + str(location))
    return audio_files
