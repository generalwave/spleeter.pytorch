from torch.utils.data import Dataset
import pandas as pd
import os
import h5py
from numpy import random
import numpy as np
from ..audio.ffmpeg import FFMPEGProcessAudioAdapter
from ..utils.stft import STFT


class MusicAnnotationTransform(object):
    def __init__(self, mix_name, instrument_list, csv):
        self.instrument_list = [mix_name] + instrument_list
        self.csv = csv
        self.annotations = self._get_annotations()

    def _get_annotations(self):
        df = pd.read_csv(self.csv)
        annotations = df.to_dict(orient='records')
        for anno in annotations:
            for key in list(anno.keys()):
                if key not in self.instrument_list:
                    del anno[key]
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __call__(self, idx):
        return self.annotations[idx]


class MusicDataset(Dataset):
    def __init__(self, mix_name, instrument_list, csv, audio_root, cache_root,
                 sample_rate, channels, frame_length, frame_step, segment_length, frequency_bins):
        self.anno_transform = MusicAnnotationTransform(mix_name, instrument_list, csv)
        self.audio_root = audio_root
        self.cache_root = cache_root
        self.sample_rate = sample_rate
        self.channels = channels
        self.segment_length = segment_length
        self.frequency_bins = frequency_bins

        self.cache_list = self._make_cache_list()
        self.adapter = FFMPEGProcessAudioAdapter()
        self.stft = STFT(frame_length, frame_step)

    def _make_cache_list(self):
        cache = []
        for idx in range(len(self.anno_transform)):
            anno = self.anno_transform(idx)
            item = dict()
            for key, value in anno.items():
                path = os.path.join(self.cache_root, value)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                item[key] = path
            cache.append(item)
        return cache

    def _generate_cache(self, anno, cache):
        spectrograms = dict()
        shape = []
        for key, value in anno.items():
            wave = self.adapter.load(path=os.path.join(self.audio_root, value),
                                     sample_rate=self.sample_rate, channels=self.channels)
            spectrogram = self.stft.stft(wave)
            spectrograms[key] = np.abs(spectrogram)
            shape.append(spectrogram.shape[0])

        shape = min(shape)

        for key in anno:
            file = h5py.File(cache[key], 'w')
            file['spectrogram'] = spectrograms[key][:shape]
            file.flush()
            file.close()

    @staticmethod
    def _has_cache(cache):
        has_cache = True
        for value in cache.values():
            if not os.path.exists(value):
                has_cache = False
                break
        return has_cache

    def __len__(self):
        return len(self.anno_transform)

    def __getitem__(self, idx):
        cache = self.cache_list[idx]

        if not self._has_cache(cache):
            anno = self.anno_transform(idx)
            self._generate_cache(anno, cache)

        spectrograms = dict()
        start, end = 0, self.segment_length
        for key, value in cache.items():
            file = h5py.File(value, 'r')
            if start == 0:
                shape = file['spectrogram'].shape[0]
                high = shape - self.segment_length
                start = random.randint(low=1, high=high)
                end = start + self.segment_length
            spectrogram = file['spectrogram'][start:end]
            spectrogram = np.transpose(spectrogram[:, :self.frequency_bins], axes=(2, 0, 1))
            spectrograms[key] = spectrogram
            file.close()

        return spectrograms
