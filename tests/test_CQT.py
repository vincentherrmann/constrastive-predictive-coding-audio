from unittest import TestCase
from constant_q_transform import *
from audio_dataset import *
from matplotlib import pyplot as plt


class TestCQT(TestCase):
    def setUp(self):
        self.cqt = CQT(sr=16000,
                       fmin=30,
                       n_bins=256,
                       bins_per_octave=32,
                       filter_scale=0.5,
                       hop_length=128,
                       trainable=False)

        self.pd = PhaseDifference(sr=16000,
                                  fmin=30,
                                  n_bins=256,
                                  bins_per_octave=32,
                                  hop_length=128)

        self.dataset = AudioDataset(
            location='/Users/vincentherrmann/Documents/Projekte/Immersions/MelodicProgressiveHouse_Tracks_small_test',
            item_length=64000)

    def test_PhaseDiff(self):
        test_data = self.dataset[0].view(1, 1, -1)
        cqt = self.cqt(test_data)
        scal = torch.log(abs(cqt)**2 + 1e-9)
        phase = angle(cqt)
        phase_diff = self.pd(phase)
        plt.imshow(scal.squeeze(), origin='lower')
        plt.imshow(phase_diff.squeeze(), origin='lower')
        pass
