from unittest import TestCase
from audio_dataset import *

import time
import torchaudio


class TestAudioDataset(TestCase):
    def test_dataset(self):
        dataset = AudioDataset(location='/Users/vincentherrmann/Documents/Projekte/Immersions/audio_clips/dataset',
                               item_length=500)
        print("dataset has length", len(dataset))
        assert len(dataset) == 531

        sample = dataset[0]
        assert sample.shape[0] == 500

        sample = dataset[530]
        assert sample.shape[0] == 500

    def test_dataset_with_unique_length(self):
        dataset = AudioDataset(location='/Users/vincentherrmann/Documents/Projekte/Immersions/audio_clips/dataset',
                               item_length=500,
                               unique_length=267)
        print("dataset has length", len(dataset))
        assert len(dataset) == 996

        sample = dataset[0]
        assert sample.shape[0] == 500

        sample = dataset[995]
        assert sample.shape[0] == 500

    def test_minibatch_performance(self):
        dataset = AudioDataset(location='/Users/vincentherrmann/Documents/Projekte/Immersions/MelodicProgressiveHouse_Tracks_test',
                               item_length=12465)

        num_workers = 2
        num_batches = 100

        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=32,
                                                 shuffle=False,
                                                 num_workers=num_workers)
        dataloader_iter = iter(dataloader)

        def calc_batches(num=1):
            for i in range(num):
                mb = next(dataloader_iter)
            return mb

        print('start loading')
        tic = time.time()
        last_minibatch = calc_batches(num_batches)
        toc = time.time()

        time_per_minibatch = (toc-tic) / num_batches

        print("time per minibatch:", time_per_minibatch, "s with", num_workers, "workers")
        assert False

        # time per minibatch: 0.8968299508094788 s with 8 workers
        # time per minibatch: 0.3775998401641846 s with 8 workers
        # time per minibatch: 0.33037535190582273 s with 4 workers
        # time per minibatch: 1.2090770196914673 s with 1 workers
        # time per minibatch: 0.3904983305931091 s with 16 workers

        # using torchaudio:
        # time per minibatch: 0.003301420211791992 s with 4 workers
        # one external storage:
        # time per minibatch: 0.01412742853164673 s with 4 workers

        # using large files:
        # time per minibatch: 0.04821853160858154 s with 2 workers
        # time per minibatch: 0.046169569492340086 s with 4 workers
        # time per minibatch: 0.07003917932510376 s with 16 workers
