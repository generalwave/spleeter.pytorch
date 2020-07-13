from .utils.stft import STFT
import numpy as np
import torch


class Separator(object):
    def __init__(self, pretrain_model, instruments, frame_length, frame_step,
                 segment_length, frequency_bins, separation_exponent, mask_extension):
        self.pretrain_model = pretrain_model
        self.instruments = instruments
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.segment_length = segment_length
        self.frequency_bins = frequency_bins
        self.separation_exponent = separation_exponent
        self.mask_extension = mask_extension

        self.model = self._get_pretrain_model()
        self.stft = STFT(frame_length, frame_step)
        self.eps = 1e-10

    def _get_pretrain_model(self):
        model = torch.load(self.pretrain_model, map_location='cpu')
        model.eval()
        return model

    def _pad_and_partition(self, data, axis):
        split_size = self.segment_length
        padding = split_size - np.mod(data.shape[0], split_size)
        pad_width = [[0, 0]] * len(data.shape)
        pad_width[axis] = [0, padding]
        data = np.pad(data, pad_width)
        split_num = data.shape[0] // split_size
        data = data.reshape((split_num, split_size) + data.shape[1:])
        return data

    def _preprocess(self, wave):
        spectrogram = self.stft.stft(wave)
        # 全部扩展或者截断为双通道数据，最好在解码时搞成双通道最佳
        if len(spectrogram) == 1:
            reps = [2] + [1] * (len(spectrogram.shape) - 1)
            spectrogram = np.tile(spectrogram, reps=reps)
        else:
            spectrogram = spectrogram[..., :2]
        src_stft = spectrogram.copy()
        spectrogram = self._pad_and_partition(spectrogram, axis=0)
        spectrogram = np.abs(spectrogram)
        spectrogram = spectrogram[:, :, :self.frequency_bins, :]
        spectrogram = np.ascontiguousarray(np.transpose(spectrogram, axes=[0, 3, 1, 2]))
        spectrogram = torch.from_numpy(spectrogram)
        return spectrogram, src_stft

    def _extend_mask(self, mask):
        extension = self.mask_extension
        if extension == 'average':
            value = mask.mean(dim=-1)
        elif extension == 'zeros':
            value = torch.zeros(mask.shape[:-1], dtype=mask.dtype, device=mask.device).unsqueeze(dim=-1)
        else:
            raise ValueError('不支持该扩展方式')
        row = self.frame_length // 2 + 1 - self.frequency_bins
        value = torch.repeat_interleave(value, repeats=row, dim=-1)
        mask = torch.cat((mask, value), dim=-1)
        return mask

    def _postprocess(self, spectrogram, masks, length):
        # 计算混合后的和
        denominator = 0
        for instrument in self.instruments:
            denominator = torch.pow(masks[instrument], self.separation_exponent) + denominator
        denominator = denominator + self.eps

        outputs = {}
        for instrument in self.instruments:
            mask = masks[instrument]
            mask = (torch.pow(mask, self.separation_exponent) + (self.eps / len(masks))) / denominator
            mask = self._extend_mask(mask)
            mask = mask.permute(dims=[0, 2, 3, 1])
            mask = mask.reshape(shape=((-1, ) + mask.shape[2:]))
            mask = mask.numpy()
            # 因为预处理可能增加padding，这里砍掉
            mask = mask[:spectrogram.shape[0], ...]

            output = mask * spectrogram
            output = self.stft.istft(output, length)
            outputs[instrument] = output

        return outputs

    def separate(self, wave):
        spectrogram, src_stft = self._preprocess(wave)

        with torch.no_grad():
            masks = self.model(spectrogram)

        waves = self._postprocess(src_stft, masks, len(wave))

        return waves
