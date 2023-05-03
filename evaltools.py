from BaseFun import BaseFun
import numpy as np
import cv2 as cv
import os

test_tool = BaseFun()

def eval_img(img1, img2):
    psnr = test_tool._psnr(img1, img2)
    ssim = test_tool._ssim(img1, img2)
    return psnr, ssim

def get_show_img(img1, img2, dim=2, slice=90):
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    img1 = np.where(img1 > 1, 1, img1)
    img2 = np.where(img2 > 1, 1, img2)
    img1 = img1 * 255
    img2 = img2 * 255
    img1 = img1.astype('uint8')
    img2 = img2.astype('uint8')
    show_img = None
    if dim == 0:
        show_img = np.concatenate((img1[slice, :, :], img2[slice, :, :]), axis=1)
    elif dim == 1:
        show_img = np.concatenate((img1[:, slice, :], img2[:, slice, :]), axis=1)
    elif dim == 2:
        show_img = np.concatenate((img1[:, :, slice], img2[:, :, slice]), axis=1)
    return show_img

if __name__ == '__main__':
    # file_path = 'results'
    file_path = r'C:\Users\ChxxxXL\Documents\MedicalImage\result'
    file_lists = [['final_imgcgan_grad_img.npy', 'gt_imgcgan_grad_img.npy'], 
                ['final_imgcgan_grad_loss.npy', 'gt_imgcgan_grad_loss.npy'],
                ['final_imgcgan.npy', 'gt_imgcgan.npy'],
                ['final_imgunet_same_img.npy', 'gt_imgunet_same_img.npy']]
    for files in file_lists:
        img1 = np.load(os.path.join(file_path, files[0]))
        img2 = np.load(os.path.join(file_path, files[1]))
        # print(img1.shape)
        print('test {} start!!'.format(files))
        psnr_total, ssim_total = eval_img(img1, img2)
        for idx, (i1, i2) in enumerate(zip(img1, img2)):
            show_img = get_show_img(i1[0], i2[0])
            psnr, ssim = eval_img(i1[0], i2[0])
            print('PSNR:{}, SSIM:{}'.format(psnr, ssim))
            cv.imwrite(os.path.join(file_path, files[0].replace('.npy', '') + '_show_img{}.jpg'.format(idx)), show_img)
        print('Total PSNR:{}, Total SSIM:{}'.format(psnr_total, ssim_total))
