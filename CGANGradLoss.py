import torch.nn.functional as F
import torch
import numpy as np
from loss.gan_loss import GenGradLoss, DscLoss
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter

from CGANTrainer import CGANTrainer


class CGANGradLoss(CGANTrainer):

    def __init__(self, args, network, dsc_network):
        super().__init__(args, network, dsc_network)
    
    def train(self, datasets):
        writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.model_name+'_train'))
        # 加载训练集 batch_size 设为 1
        train_loader = DataLoader(datasets, batch_size=self.batch_size, shuffle=True)

        # 继续训练
        if self.load_model:
            self.load()

        # 设置网络和优化器
        g_opt = self._optimizer(self.network)
        d_opt = self._dsc_optimizer(self.dsc_network)
        gen_loss_fun = GenGradLoss()
        dsc_loss_fun = DscLoss()

        if self.gpu_mode:
            self.network = self.network.cuda()
            self.dsc_network = self.dsc_network.cuda()
            gen_loss_fun = gen_loss_fun.cuda()
            dsc_loss_fun = dsc_loss_fun.cuda()

        if torch.cuda.device_count() > 1:
            self.network = torch.nn.DataParallel(self.network)
            self.dsc_network = torch.nn.DataParallel(self.dsc_network)

        ones, zeros = None, None
        first_iter = True
        step = 0
        self.dsc_network.train()
        print('start training!!! time: {}'.format(self.get_local_time()))
        for epoch in range(self.epoch):
            self.network.train()
            for batch_id, data in enumerate(train_loader):
                d_opt.zero_grad()
                x_data = data[0].cuda()
                y_data = data[1].cuda()
                predicts = self.network(x_data)
                real_data = torch.cat((x_data, y_data), dim=1)
                fake_data = torch.cat((x_data, predicts), dim=1)
                d_real = self.dsc_network(real_data)
                d_fake = self.dsc_network(fake_data)
                if first_iter:
                    first_iter = False
                    ones = torch.ones_like(d_real)
                    zeros = torch.zeros_like(d_fake)
                d_loss = dsc_loss_fun(d_real, d_fake, zeros, ones)
                d_loss.backward()
                d_opt.step()
                
                g_opt.zero_grad()
                predicts = self.network(x_data)
                fake_data = torch.cat((x_data, predicts), dim=1)
                g_fake = self.dsc_network(fake_data)
                g_loss = gen_loss_fun(g_fake, predicts, y_data, zeros)
                g_loss.backward()
                g_opt.step()

                step += 1
                
                if batch_id % 100 == 0:
                    print("epoch: {}, batch_id: {}, g_loss is: {}, d_loss is: {}, acc is: {}".format(epoch, batch_id, g_loss.item(), d_loss.item(), 0))
                    writer.add_scalar('g_loss', g_loss.item(), epoch * len(train_loader) + step)
                    writer.add_scalar('d_loss', d_loss.item(), epoch * len(train_loader) + step)
                if (batch_id + 1) % 100 == 0:
                    self.save()
            # 每一轮结束，保存模型
            self.save()
            print('epoch {} finish, time: {}'.format(epoch, self.get_local_time()))
        print('train finish!!! time: {}'.format(self.get_local_time()))
        writer.close()
