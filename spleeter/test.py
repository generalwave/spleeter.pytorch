from .audio.ffmpeg import FFMPEGProcessAudioAdapter
from .separator import Separator
from .config import CONFIG


def main():
    audio = FFMPEGProcessAudioAdapter()
    wave = audio.load('/Users/yangjiang/temp/video/data/mix/test.flac')
    pretrain_model = '/Users/yangjiang/temp/gpu1/SSD_epoch_399_1.5179922580718994.pth'
    instrument_list = CONFIG['instrument_list']
    frame_length = CONFIG['frame_length']
    frame_step = CONFIG['frame_step']
    segment_length = CONFIG['segment_length']
    frequency_bins = CONFIG['frequency_bins']
    separation_exponent = 1
    mask_extension = 'zeros'
    separator = Separator(pretrain_model, instrument_list, frame_length, frame_step,
                          segment_length, frequency_bins, separation_exponent, mask_extension)
    results = separator.separate(wave)

    for key, value in results.items():
        audio.save(f'/Users/yangjiang/temp/video/{key}.wav', value, 44100, 2, 'wav', '128k')
