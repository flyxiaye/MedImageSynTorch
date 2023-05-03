import argparse, os, torch
from networks.cgan import Gen, Dsc
import networks.unet as unet
import networks.srcgan as srcgan
from datasets.ImgDataset import ImgDataset
from CGANTrainer import CGANTrainer
from CGANGradLoss import CGANGradLoss
from CGANGradImg import CGANGradImg
from UnetTrainer import UNetTrainer
from SRCGAN import SRCGAN

import numpy as np
# from GAN import GAN
# from CGAN import CGAN
# from LSGAN import LSGAN
# from DRAGAN import DRAGAN
# from ACGAN import ACGAN
# from WGAN import WGAN
# from WGAN_GP import WGAN_GP
# from infoGAN import infoGAN
# from EBGAN import EBGAN
# from BEGAN import BEGAN

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'train_test'], help='which mode to select')
    parser.add_argument('--model_type', type=str, default='GAN',
                        choices=['cgan', 'cgan_grad_loss', 'cgan_grad_img', 'unet', 'unet_same_img','srcgan', 'res_srcgan'],
                        help='The type of model')
    parser.add_argument('--data_dir', type=str, default='~/dealedMRCT',
                        help='The dir of dataset')
    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=2, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=128, help='The size of input image')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)
    parser.add_argument('--load_model', action='store_true')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

        # declare instance for GAN
    if args.model_type == 'cgan':
        gan = CGANTrainer(args, Gen(), Dsc())
    elif args.model_type == 'cgan_grad_loss':
        gan = CGANGradLoss(args, Gen(), Dsc())
    elif args.model_type == 'cgan_grad_img':
        gan = CGANGradImg(args, Gen(), Dsc(3))
    elif args.model_type == 'unet':
        gan = UNetTrainer(args, unet.Gen())
    elif args.model_type == 'unet_same_img':
        gan = UNetTrainer(args, Gen())
    elif args.model_type == 'srcgan':
        gan = SRCGAN(args, srcgan.Gen(), srcgan.Dsc())
    elif args.model_type == 'res_srcgan':
        gan = SRCGAN(args, srcgan.Gen(1, True), srcgan.Dsc())
    else:
        raise Exception("[!] There is no option for " + args.model_type)

    if args.mode == 'train' or args.mode == 'train_test':
        # launch the graph in a session
        gan.train(ImgDataset('train.txt', args.data_dir))
        print(" [*] Training finished! epochs: {}".format(args.epoch))
    if args.mode == 'test' or args.mode == 'train_test':
        # visualize learned generator
        _, _, final_img, gt_img = gan.evaluate(ImgDataset('test.txt', args.data_dir))
        print(" [*] Testing finished!")
        np.save(os.path.join(args.result_dir, 'final_img' + args.model_type), final_img)
        np.save(os.path.join(args.result_dir, 'gt_img' + args.model_type), gt_img)

if __name__ == '__main__':
    main()