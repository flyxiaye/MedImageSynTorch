from BaseTrainer import BaseTrainer
from loss.gan_loss import GenLoss, DscLoss
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from datasets.myutiles import recompone_overlap
import numpy as np


class UNetTrainer(BaseTrainer):

    def __init__(self, args, network):
        super(UNetTrainer, self).__init__(args, network)
        # self.dsc_network = dsc_network

        # self.lambda_l1 = 100

    def train(self, datasets):
        # writer = LogWriter(logdir=self.log_dir+"train")
        # 加载训练集 batch_size 设为 1
        train_loader = DataLoader(datasets, batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=True)
        # 设置网络和优化器
        opt = self._optimizer(self.network)
        loss_fun = torch.nn.MSELoss()

        # 继续训练
        if self.load_model:
            self.load()

        if self.gpu_mode:
            self.network = self.network.cuda()
            loss_fun = loss_fun.cuda()

        if torch.cuda.device_count() > 1:
            self.network = torch.nn.DataParallel(self.network)

        step = 0
        print('start training!!! time: {}'.format(self.get_local_time()))
        for epoch in range(self.epoch):
            self.network.train()
            for batch_id, data in enumerate(train_loader):
                opt.zero_grad()
                x_data = data[0].cuda()
                y_data = data[1].cuda()
                predicts = self.network(x_data)

                loss = loss_fun(predicts, y_data)
                loss.backward()
                opt.step()

                step += 1
                
                if batch_id % 100 == 0:
                    print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.item(), 0))
                if (batch_id + 1) % 100 == 0:
                    self.save()
            # 每一轮结束，保存模型
            self.save()
            print('epoch {} finish, time: {}'.format(epoch ,self.get_local_time()))
        print('train finish!!! time: {}'.format(self.get_local_time()))


    def _optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

