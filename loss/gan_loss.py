import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GenLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, dsc_fake, predicts, y_data, zeros):
        dsc_fake_loss = F.binary_cross_entropy_with_logits(dsc_fake, zeros)
        g_loss_l1 = F.l1_loss(y_data, predicts)
        return dsc_fake_loss + g_loss_l1 * 100

class DscLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, dsc_fake, dsc_real, zeros, ones):
        d_real_loss = F.binary_cross_entropy_with_logits(dsc_real, ones)
        d_fake_loss = F.binary_cross_entropy_with_logits(dsc_fake, zeros)
        return d_real_loss + d_fake_loss

class GenGradLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, dsc_fake, predicts, y_data, zeros):
        dsc_fake_loss = F.binary_cross_entropy_with_logits(dsc_fake, zeros)
        g_loss_l1 = F.l1_loss(y_data, predicts)
        input = y_data.cpu().detach().numpy()
        label = predicts.cpu().detach().numpy()
        grad_x = F.mse_loss(torch.Tensor(np.abs(np.gradient(input, axis=2))), torch.Tensor(np.abs(np.gradient(label, axis=2))))
        grad_y = F.mse_loss(torch.Tensor(np.abs(np.gradient(input, axis=3))), torch.Tensor(np.abs(np.gradient(label, axis=3))))
        grad_z = F.mse_loss(torch.Tensor(np.abs(np.gradient(input, axis=4))), torch.Tensor(np.abs(np.gradient(label, axis=4))))
        g_loss_gd = (grad_x + grad_y + grad_z).cuda()
        return dsc_fake_loss + g_loss_l1 * 100 + 100 * g_loss_gd

class GenGradImgLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dsc_fake, predicts, y_data, zeros, pred_grads, y_data_grads):
        dsc_fake_loss = F.binary_cross_entropy_with_logits(dsc_fake, zeros)
        g_loss_l1 = F.l1_loss(y_data, predicts)
        g_loss_l1_grad = F.l1_loss(y_data_grads, pred_grads)
        return dsc_fake_loss + 100 * g_loss_l1 + 100 * g_loss_l1_grad
