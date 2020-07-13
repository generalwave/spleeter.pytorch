import musdb
import museval
from glob import glob
import os
import json
import numpy as np
from .audio.ffmpeg import FFMPEGProcessAudioAdapter
from .separator import Separator
from .config import CONFIG


def separate(musdb_root, subset, pretrain_model, result_root):
    audio = FFMPEGProcessAudioAdapter()
    instrument_list = CONFIG['instrument_list']
    frame_length = CONFIG['frame_length']
    frame_step = CONFIG['frame_step']
    segment_length = CONFIG['segment_length']
    frequency_bins = CONFIG['frequency_bins']
    separation_exponent = 2
    mask_extension = 'zeros'
    separator = Separator(pretrain_model, instrument_list, frame_length, frame_step,
                          segment_length, frequency_bins, separation_exponent, mask_extension)

    songs = glob(os.path.join(musdb_root, subset, '*/mixture.wav'))

    for song in songs:
        wave = audio.load(song)
        results = separator.separate(wave)

        foldername = os.path.basename(os.path.dirname(song))
        for key, value in results.items():
            out_path = os.path.join(result_root, subset, foldername, f'{key}.wav')
            audio.save(out_path, value, 44100, 2, 'wav', '128k')


def compute_musdb_metrics(musdb_root, subset, result_root, metrics_root, instruments, metrics):
    dataset = musdb.DB(root=musdb_root, is_wav=True, subsets=[subset])
    museval.eval_mus_dir(dataset=dataset, estimates_dir=result_root, output_dir=metrics_root)

    songs = glob(os.path.join(metrics_root, subset, '*.json'))
    metrics = {instrument: {k: [] for k in metrics} for instrument in instruments}
    for song in songs:
        with open(song, 'r') as stream:
            data = json.load(stream)
        for target in data['targets']:
            instrument = target['name']
            for metric in metrics:
                sdr_med = np.median([
                    frame['metrics'][metric]
                    for frame in target['frames']
                    if not np.isnan(frame['metrics'][metric])])
                metrics[instrument][metric].append(sdr_med)
    return metrics


def main():
    musdb_root = '/Users/yangjiang/temp/video/data'
    subset = 'test'
    pretrain_model = '/Users/yangjiang/temp/video/model.pth'
    result_root = '/Users/yangjiang/temp/video/output/audio'
    separate(musdb_root, subset, pretrain_model, result_root)

    metrics_root = '/Users/yangjiang/temp/video/output/metrics'
    instruments = ['vocals', 'drums', 'bass', 'other']
    metrics = ['SDR', 'SAR', 'SIR', 'ISR']
    metrics = compute_musdb_metrics(musdb_root, subset, result_root, metrics_root, instruments, metrics)

    for instrument, metric in metrics.items():
        print(instrument)
        for key, value in metric.items():
            print(key, f'{np.median(value):.3f}')
