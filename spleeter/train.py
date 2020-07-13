import os
import numpy as np
import torch
from numpy import random
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data
from time import time
from tqdm import tqdm
from .config import CONFIG
from .models.spleeternet import SpleeterNet
from .models.multi_loss import MultiLoss
from .datasets.dataset import MusicDataset
from .utils.meter import AverageMeter


def reproducible():
    seed = 2099
    import torch.backends.cudnn
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def collate_fn(batch):
    mix_spectrograms = list()
    sub_spectrograms = dict()
    for sample in batch:
        for key in sample:
            if key == CONFIG['mix_name']:
                mix_spectrograms.append(sample[key])
            else:
                if key in sub_spectrograms:
                    sub_spectrograms[key].append(sample[key])
                else:
                    sub_spectrograms[key] = [sample[key]]

    mix_spectrograms = torch.from_numpy(np.array(mix_spectrograms))
    for key in sub_spectrograms:
        sub_spectrograms[key] = torch.from_numpy(np.array(sub_spectrograms[key]))

    return mix_spectrograms, sub_spectrograms


def get_data_loader():
    # 训练配置参数
    batch_size = CONFIG['batch_size']
    thread_num = CONFIG['thread_num']
    # Dataset 参数
    train_csv = CONFIG['train_csv']
    val_csv = CONFIG['val_csv']
    audio_root = CONFIG['audio_root']
    cache_root = CONFIG['cache_root']
    # Dataset 基础参数
    mix_name = CONFIG['mix_name']
    instrument_list = CONFIG['instrument_list']
    sample_rate = CONFIG['sample_rate']
    channels = CONFIG['channels']
    frame_length = CONFIG['frame_length']
    frame_step = CONFIG['frame_step']
    segment_length = CONFIG['segment_length']
    frequency_bins = CONFIG['frequency_bins']

    train_dataset = MusicDataset(mix_name, instrument_list, train_csv, audio_root, cache_root,
                                 sample_rate, channels, frame_length, frame_step, segment_length, frequency_bins)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=thread_num, drop_last=True, collate_fn=collate_fn,
                                       worker_init_fn=lambda work_id: random.seed(torch.initial_seed() & 0xffffffff))

    val_dataset = MusicDataset(mix_name, instrument_list, val_csv, audio_root, cache_root,
                               sample_rate, channels, frame_length, frame_step, segment_length, frequency_bins)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=thread_num, drop_last=False, collate_fn=collate_fn,
                                     worker_init_fn=lambda work_id: random.seed(torch.initial_seed() & 0xffffffff))

    return train_dataloader, val_dataloader


def train_one_epoch(model, device, loader, optimizer, criterion):
    model.train()

    meters = dict()
    meters['loss'] = AverageMeter()
    meters.update({key: AverageMeter() for key in CONFIG['instrument_list']})

    for mix_spectrograms, sub_spectrograms in tqdm(loader):
        batch_size = len(mix_spectrograms)
        mix_spectrograms = mix_spectrograms.to(device)
        for key in sub_spectrograms:
            sub_spectrograms[key] = sub_spectrograms[key].to(device)

        predict = model(mix_spectrograms)

        optimizer.zero_grad()
        loss, sub_loss = criterion(predict, sub_spectrograms)
        loss.backward()
        optimizer.step()

        meters['loss'].update(loss.item(), batch_size)
        for key in sub_loss:
            meters[key].update(sub_loss[key].item(), batch_size)

    return meters


def val_one_epoch(model, device, loader,  criterion):
    model.eval()

    meters = dict()
    meters['loss'] = AverageMeter()
    meters.update({key: AverageMeter() for key in CONFIG['instrument_list']})

    with torch.no_grad():
        for mix_spectrograms, sub_spectrograms in loader:
            batch_size = len(mix_spectrograms)
            mix_spectrograms = mix_spectrograms.to(device)
            for key in sub_spectrograms:
                sub_spectrograms[key] = sub_spectrograms[key].to(device)

            predict = model(mix_spectrograms)

            loss, sub_loss = criterion(predict, sub_spectrograms)

            meters['loss'].update(loss.item(), batch_size)
            for key in sub_loss:
                meters[key].update(sub_loss[key].item(), batch_size)

    return meters


def train(model, device):
    # 训练配置参数
    max_epoch = CONFIG['max_epoch']
    lr = CONFIG['lr']
    weight_decay = CONFIG['weight_decay']
    # 学习率调整参数
    milestones = CONFIG['milestones']
    gamma = CONFIG['gamma']
    # 损失参数
    instrument_list = CONFIG['instrument_list']
    instrument_weight = CONFIG['instrument_weight']
    # 模型保存路径
    save_directory = CONFIG['save_directory']
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = MultiLoss(instrument_list, instrument_weight)
    scheduler = MultiStepLR(optimizer, milestones, gamma)

    train_loader, val_loader = get_data_loader()

    for i in range(max_epoch):
        start = time()
        t_loss = train_one_epoch(model, device, train_loader, optimizer, criterion)
        if val_loader is not None:
            v_loss = val_one_epoch(model, device, val_loader, criterion)
        else:
            v_loss = t_loss
        end = time()

        scheduler.step()

        msg = ''
        for key, value in t_loss.items():
            value = value.result()
            msg += f'{key}:{value:.4f}\t'
        for key, value in v_loss.items():
            value = value.result()
            msg += f'{key}:{value:.4f}\t'
        msg += f'time:{(end - start):.1f}\tepoch:{i}'
        print(msg)

        save_path = os.path.join(save_directory, 'SSD_epoch_' + str(i) + '_' + str(v_loss['loss'].result()) + '.pth')
        model.phase = 'test'
        torch.save(model, save_path)
        model.phase = 'train'


def main():
    reproducible()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    instruments = CONFIG['instrument_list']
    output_mask_logit = False
    keras = CONFIG['keras']
    model = SpleeterNet(instruments, output_mask_logit, phase='train', keras=keras)
    model.to(device)

    pretrain_model = CONFIG['pretrain_model']
    if pretrain_model is not None:
        model.load_state_dict(torch.load(pretrain_model, map_location=device))

    train(model, device)
