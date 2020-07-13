from torch.nn import functional


def _make_weights_dict(instruments, weights):
    result = dict()
    for name, weight in zip(instruments, weights):
        result[name] = weight
    return result


class MultiLoss:
    def __init__(self, instruments, weights):
        super(MultiLoss, self).__init__()
        self.weights = _make_weights_dict(instruments, weights)

    def __call__(self, predict, target):
        loss, sub_loss = 0, dict()

        for key in self.weights:
            weight = self.weights[key]
            cur_loss = weight * functional.l1_loss(predict[key], target[key])

            loss = loss + cur_loss
            sub_loss[key] = cur_loss

        return loss, sub_loss
