import torch
import torch.nn as nn
import torch.nn.functional as F


class reg_loss(nn.Module):
    def __init__(self):
        # super().__init__()
        super(reg_loss, self).__init__()
        self.loss = nn.MSELoss()
    def forward(self, pred, gt):
        '''
        :param pred: shape=(B, )
        :param y: shape=(B, )
        :return: loss
        '''
        if pred.dim() > gt.dim():
            pred = pred.squeeze(1)
        loss = self.loss(pred, gt)
        return loss

class LabelSmoothingCE(nn.Module):
    """
    带有标签平滑的交叉熵损失
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1 - smoothing

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        :param pred: (B, num_class, N)  分类时N=1，分割时N等于点云数量
        :param gt: (B, N)
        :return: loss, acc
        """
        B, cls, N = pred.shape

        # (B, cls, N) -> (B*N, cls)
        pred = pred.permute(0, 2, 1).reshape(B*N, cls)
        gt = gt.reshape(B*N,)

        acc = torch.sum(torch.max(pred, dim=-1)[1] == gt) / (B * N)

        logprobs = F.log_softmax(pred, dim=-1)
        loss_pos = -logprobs.gather(dim=-1, index=gt.unsqueeze(1)).squeeze(1)
        loss_smoothing = -logprobs.mean(-1)
        loss = self.confidence * loss_pos + self.smoothing * loss_smoothing

        return loss.mean(), acc.item()



