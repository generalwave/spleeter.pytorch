from spleeter.models.spleeternet import SpleeterNet
import tensorflow as tf
import numpy as np
import torch


def tfid2name(idx):
    name = '' if idx == 0 else '_' + str(idx)
    return name


def tfconv2torch(output, layer_name, tf_vars, var_name):
    output[f'{layer_name}.weight'] = np.transpose(tf_vars[f'{var_name}/kernel'], axes=(3, 2, 0, 1))
    output[f'{layer_name}.bias'] = tf_vars[f'{var_name}/bias']


def tfbn2torch(output, layer_name, tf_vars, var_name):
    output[f'{layer_name}.weight'] = tf_vars[f'{var_name}/gamma']
    output[f'{layer_name}.bias'] = tf_vars[f'{var_name}/beta']
    output[f'{layer_name}.running_mean'] = tf_vars[f'{var_name}/moving_mean']
    output[f'{layer_name}.running_var'] = tf_vars[f'{var_name}/moving_variance']


def tf2torch(checkpoint_path, instrument_list):
    # 获取变量名称
    tf_vars = {}
    init_vars = tf.train.list_variables(checkpoint_path)
    # 根据变量名称获取值
    for name, shape in init_vars:
        data = tf.train.load_variable(checkpoint_path, name)
        tf_vars[name] = data

    result = dict()
    conv_idx, deconv_idx, bn_idx = 0, 0, 0
    for instrument in instrument_list:
        # 下采样过程
        for i in range(6):
            layer_name = f'{instrument}.conv_layers.{i}'

            conv_suffix = tfid2name(conv_idx)
            conv_idx += 1
            var_name = f'conv2d{conv_suffix}'
            tfconv2torch(result, f'{layer_name}.conv', tf_vars, var_name)

            bn_suffix = tfid2name(bn_idx)
            bn_idx += 1
            var_name = f'batch_normalization{bn_suffix}'
            tfbn2torch(result, f'{layer_name}.bn', tf_vars, var_name)

        # 上采样过程
        for i in range(6):
            layer_name = f'{instrument}.deconv_layers.{i}' if i != 5 else f'{instrument}.mask_layer'

            deconv_suffix = tfid2name(deconv_idx)
            deconv_idx += 1
            var_name = f'conv2d_transpose{deconv_suffix}'
            tfconv2torch(result, f'{layer_name}.conv', tf_vars, var_name)

            bn_suffix = tfid2name(bn_idx)
            bn_idx += 1
            var_name = f'batch_normalization{bn_suffix}'
            tfbn2torch(result, f'{layer_name}.bn', tf_vars, var_name)

        conv_suffix = tfid2name(conv_idx)
        conv_idx += 1
        var_name = f'conv2d{conv_suffix}'
        tfconv2torch(result, f'{instrument}.last_layer', tf_vars, var_name)

    return result


def copy2model(model, checkpoint):
    state_dict = model.state_dict()

    for key, value in checkpoint.items():
        if key in state_dict:
            target_shape = state_dict[key].shape
            assert target_shape == value.shape
            state_dict.update({key: torch.from_numpy(value)})
        else:
            raise ValueError('导出错误')

    model.load_state_dict(state_dict)


def main():
    instrument_list = ['vocals', 'accompaniment']
    model = SpleeterNet(instrument_list, output_mask_logit=False, phase='test', keras=True)

    checkpoint = tf2torch('/Users/yangjiang/Downloads/2stems', instrument_list)
    copy2model(model, checkpoint)
    torch.save(model, '/Users/yangjiang/temp/video/model.pth')


def test():
    model = torch.load('/Users/yangjiang/temp/video/model.pth')
    model.eval()
    x = np.load('/Users/yangjiang/temp/video/input.npy', allow_pickle=True)
    x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))
    x = torch.from_numpy(x)
    y = model(x)
    print(y)


if __name__ == '__main__':
    main()
    test()
