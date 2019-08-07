cqt_default_dict = {'sample_rate': 16000,
                    'fmin': 30,
                    'n_bins': 256,
                    'bins_per_octave': 32,
                    'filter_scale': 0.5,
                    'hop_length': 128,
                    'trainable_cqt': False}

cqt_high_res_dict = cqt_default_dict.copy()
cqt_high_res_dict['sample_rate'] = 44100
cqt_high_res_dict['n_bins'] = 292
cqt_high_res_dict['hop_length'] = 256
