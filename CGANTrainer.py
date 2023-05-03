from BaseTrainer import BaseTrainer
from loss.gan_loss import GenLoss, DscLoss
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from datasets.myutiles import recompone_overlap
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class CGANTrainer(BaseTrainer):

    def __init__(self, args, network, dsc_network):
        super(CGANTrainer, self).__init__(args, network)
        self.dsc_network = dsc_network

        self.lambda_l1 = 100

    def train(self, datasets):
        writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.model_name+'_train'))
        # 加载训练集 batch_size 设为 1
        train_loader = DataLoader(datasets, batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=True)
        # 设置网络和优化器
        g_opt = self._optimizer(self.network)
        d_opt = self._dsc_optimizer(self.dsc_network)
        gen_loss_fun = GenLoss()
        dsc_loss_fun = DscLoss()

        # 继续训练
        if self.load_model:
            self.load()

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
            print('epoch {} finish, time: {}'.format(epoch ,self.get_local_time()))
        print('train finish!!! time: {}'.format(self.get_local_time()))
        writer.close()

    def evaluate(self, datasets):
        with torch.no_grad():
            self.load()
            self.network.eval()
            if self.gpu_mode:
                self.network.cuda()
            test_loader = DataLoader(datasets, batch_size=1, shuffle=False)
            preds, gts = [], []
            for batch_id, data in enumerate(test_loader):
                x_ = data[0].cuda()
                y_ = data[1]
                out = self.network(x_)
                preds.append(out.cpu().numpy()[0])
                gts.append(y_.numpy()[0])
            preds = np.asarray(preds)
            gts = np.asarray(gts)
            print(preds.shape)
            print(gts.shape)
            psnrs, ssims = [], []
            for pred, gt in zip(preds, gts):
                psnrs.append(self._psnr(pred, gt))
                ssims.append(self._ssim(pred, gt))
            np.savetxt(os.path.join(self.result_dir, self.model_name+'_psnrs.txt'), psnrs)
            np.savetxt(os.path.join(self.result_dir, self.model_name+'_ssims.txt'), ssims)
            pred = recompone_overlap(preds, self.img_shape, self.stride_shape)
            gt = recompone_overlap(gts, self.img_shape, self.stride_shape)
            psnr = self._psnr(pred, gt)
            ssim = self._ssim(pred, gt)
            print("PSNR: {}, SSIM: {}".format(psnr, ssim))
            return psnr, ssim, pred, gt

    def _optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def _dsc_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def _loss(self):
        def gen_loss(dsc_fake=None, dsc_real=None, predicts=None, y_data=None, zeros=None, ones=None):
            dsc_fake_loss = F.binary_cross_entropy_with_logits(dsc_fake, zeros)
            g_loss_l1 = F.l1_loss(y_data, predicts)
            return dsc_fake_loss + g_loss_l1 * self.lambda_l1
        return gen_loss

    def _dsc_loss(self):
        def dsc_loss(dsc_fake=None, dsc_real=None, predicts=None, y_data=None, zeros=None, ones=None):
            d_real_loss = F.binary_cross_entropy_with_logits(dsc_real, ones)
            d_fake_loss = F.binary_cross_entropy_with_logits(dsc_fake, zeros)
            return d_real_loss + d_fake_loss
        return dsc_loss

    def save(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.network.state_dict(), os.path.join(save_dir, self.model_name + '_G.pth'))
        torch.save(self.dsc_network.state_dict(), os.path.join(save_dir, self.model_name + '_D.pth'))


    def load(self):
        save_dir = os.path.join(self.save_dir, self.model_name)
        if torch.cuda.device_count() > 1:
            self.network = torch.nn.DataParallel(self.network)
            self.dsc_network = torch.nn.DataParallel(self.dsc_network)
        self.network.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pth')))
        self.dsc_network.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pth')))