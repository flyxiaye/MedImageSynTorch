import sys 
# sys.path.append('/home/aistudio/external-libraries')
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.myutiles import recompone_overlap
import numpy as np
import os
# import nibabel as nib
import time
from BaseFun import BaseFun


class BaseTrainer(BaseFun):

    def __init__(self, args, network):
        super().__init__()
        self.network = network
        # self.data_dir = args.
        self.save_dir = args.save_dir
        self.model_name = args.model_type
        self.log_dir = args.log_dir


        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.gpu_mode = args.gpu_mode
        self.input_size = args.input_size
        self.load_model = args.load_model

        self.patch_shape = (128, 128, 128)
        self.img_shape = (181, 216, 181)
        self.stride_shape = (48, 48, 48)

    
    def train(self):
        pass
    
    def predict(self):
        pass
        # model = paddle.Model(self.network)
        # model.prepare(
        #     self._optimizer(model)
            
        # )
        # model.load(self.model_dir + self.model_name)
        # preds = model.predict(self.test_datasets, batch_size=1)
        # return preds
        

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
            np.savetxt(os.path.join(self.result_dir, self.model_name+'_psnrs'), psnrs)
            np.savetxt(os.path.join(self.result_dir, self.model_name+'_ssims'), ssims)
            pred = recompone_overlap(preds, self.img_shape, self.stride_shape)
            gt = recompone_overlap(gts, self.img_shape, self.stride_shape)
            psnr = self._psnr(pred, gt)
            ssim = self._ssim(pred, gt)
            print("PSNR: {}, SSIM: {}".format(psnr, ssim))
            return psnr, ssim, pred, gt

    def _optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=0.0002)

    def _loss(self):
        return F.mse_loss;

    # def _psnr(self, img1, img2, max_pixel=1):
    #     img1, img2 = np.array(img1, dtype='float'), np.array(img2, dtype='float')
    #     mse = np.mean((img1 - img2) ** 2)
    #     psnr = 10 * np.log10(max_pixel * max_pixel / mse)
    #     return psnr

    # # SSIM
    # def _ssim(self, img1, img2, K=(0.01, 0.03), L=1):
    #     C1, C2 = (K[0] * L) ** 2, (K[1] * L) ** 2
    #     C3 = C2 / 2
    #     img1, img2 = np.array(img1, dtype='float'), np.array(img2, dtype='float')
    #     m1, m2 = np.mean(img1), np.mean(img2)
    #     s1, s2 = np.std(img1), np.std(img2)
    #     s12 = np.mean((img1 - m1) * (img2 - m2))
    #     l = (2 * m1 * m2 + C1) / (m1**2 + m2**2 + C1)
    #     c = (2 * s1 * s2 + C2) / (s1**2 + s2**2 + C2)
    #     s = (s12 + C3) / (s1 * s2 + C3)
    #     return l * c * s

    def save(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.network.state_dict(), os.path.join(save_dir, self.model_name + '.pth'))
        # torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pth'))


    def load(self):
        save_dir = os.path.join(self.save_dir, self.model_name)
        if torch.cuda.device_count() > 1:
            self.network = torch.nn.DataParallel(self.network)
        self.network.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '.pth')))
        # self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

    def get_local_time(self):
        return time.asctime( time.localtime(time.time()) )