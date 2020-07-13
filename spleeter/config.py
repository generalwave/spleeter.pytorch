import os


# HOME = '/Users/yangjiang/temp/video/'
HOME = '/data1/jiang.yang/output/spleeter'


_4stems_config = {
    # 基础参数
    'max_epoch': 400,
    'lr': 1e-3,
    'milestones': [50, 150, 250, 300],
    'gamma': 0.1,
    'weight_decay': 5e-4,
    'batch_size': 16,
    'thread_num': 24,
    # 配置参数
    'mix_name': 'mix',
    'instrument_list': ['vocals', 'drums', 'bass', 'other'],
    'instrument_weight': [1, 1, 1, 1],
    'sample_rate': 44100,
    'channels': 2,
    'frame_length': 4096,
    'frame_step': 1024,
    'segment_length': 512,
    'frequency_bins': 1024,
    # 训练数据集
    'audio_root': os.path.join(HOME, 'musdb18hq'),
    'train_csv': os.path.join(HOME, 'musdb_train.csv'),
    'val_csv': os.path.join(HOME, 'musdb_validation.csv'),
    'cache_root': os.path.join(HOME, 'cache'),

    # 模型保存位置
    'save_directory': os.path.join(HOME, 'model'),
    'pretrain_model': None,
    'keras': False,
}


CONFIG = _4stems_config
