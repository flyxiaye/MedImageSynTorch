import os
from myutiles import *
import SimpleITK as sitk
import numpy as np

def prepare_data(data_dir, patch_shape=(128, 128, 128), stride_shape=(48, 48, 48)):
    dirs = ['trainMR', 'trainCT', 'testMR', 'testCT']
    for d in dirs:
        path = os.path.join(data_dir, d)
        try:
            os.mkdir(path)
        except:
            pass

    train_mr_files, train_ct_files = [], []
    test_mr_files, test_ct_files = [], []
    for f in os.listdir(data_dir):
        if 'MR.nii' in f:
            if f in ['064MR.nii', '063MR.nii', '062MR.nii', '061MR.nii']:
                test_mr_files.append(f)
                test_ct_files.append(f[:3]+'CT.nii')
            else:
                train_mr_files.append(f)
                train_ct_files.append(f[:3]+'CT.nii')
    _pre_data(train_mr_files, train_ct_files, data_dir, patch_shape, stride_shape)
    _pre_data(test_mr_files, test_ct_files, data_dir, patch_shape, stride_shape, mode='test')


def _pre_data(mr_files, ct_files, data_dir, patch_shape, stride_shape, mode='train'):
    mode = mode.lower()
    assert mode in ['train', 'test', 'predict'], \
            "mode should be 'train' or 'test' or 'predict', but got {}".format(mode)
    if mode == 'train':
        mr_img_dir = 'trainMR'
        ct_img_dir = 'trainCT'
        img_list = os.path.join(data_dir, 'train.txt')
    elif mode == 'test':
        mr_img_dir = 'testMR'
        ct_img_dir = 'testCT'
        img_list = os.path.join(data_dir, 'test.txt')
    f = open(img_list, 'w')
    idx = 0
    for mr_f, ct_f in zip(mr_files, ct_files):
        mr_f, ct_f = os.path.join(data_dir, mr_f), os.path.join(data_dir, ct_f)
        mr_img = sitk.ReadImage(mr_f)
        ct_img = sitk.ReadImage(ct_f)
        mr_img = sitk.GetArrayFromImage(mr_img)
        ct_img = sitk.GetArrayFromImage(ct_img)
        # mr_img, ct_img = nib.load(mr_f), nib.load(ct_f)
        # mr_img, ct_img = np.asarray(mr_img.dataobj, dtype='float32'), np.asarray(ct_img.dataobj, 'float32')
        # ct_img = rm_nan_ct(ct_img)
        # ct_img = rm_neg(ct_img)
        # mr_img, ct_img = rm_max(mr_img), rm_max(ct_img)
        mr_img = mr_img / 255
        ct_img = ct_img / 255

        for mr_sub_img, ct_sub_img in zip(
            extract_ordered_overlap(mr_img, patch_shape, stride_shape),
            extract_ordered_overlap(ct_img, patch_shape, stride_shape)
            ):
            mr_name = "{}/mr{}".format(mr_img_dir, idx)
            ct_name = "{}/ct{}".format(ct_img_dir, idx)
            f.write(mr_name + '\t' + ct_name + '\n')
            mr_name = os.path.join(data_dir, mr_name)
            ct_name = os.path.join(data_dir, ct_name)
            np.save(mr_name, mr_sub_img.astype('float32'))
            np.save(ct_name, ct_sub_img.astype('float32'))
            idx += 1


def prepare_data_unet(src_data_dir, dst_data_dir, input_patch_shape=(128, 128, 128), output_patch_shape=(36, 36, 36),stride_shape=(24, 24, 24)):
    dirs = ['trainMR', 'trainCT', 'testMR', 'testCT']
    for d in dirs:
        path = os.path.join(dst_data_dir, d)
        if not os.path.exists(path):
            os.makedirs(path)
    train_mr_files, train_ct_files = [], []
    test_mr_files, test_ct_files = [], []
    for f in os.listdir(src_data_dir):
        if 'MR.nii' in f:
            if f in ['064MR.nii', '063MR.nii', '062MR.nii', '061MR.nii']:
                test_mr_files.append(f)
                test_ct_files.append(f[:3]+'CT.nii')
            else:
                train_mr_files.append(f)
                train_ct_files.append(f[:3]+'CT.nii')
    _pre_data_unet(train_mr_files, train_ct_files, src_data_dir, dst_data_dir, input_patch_shape, output_patch_shape, stride_shape)
    _pre_data_unet(test_mr_files, test_ct_files, src_data_dir, dst_data_dir, input_patch_shape, output_patch_shape, stride_shape, mode='test')


def _pre_data_unet(mr_files, ct_files, src_data_dir, dst_data_dir, input_patch_shape, output_patch_shape, stride_shape, mode='train'):
    mode = mode.lower()
    assert mode in ['train', 'test', 'predict'], \
            "mode should be 'train' or 'test' or 'predict', but got {}".format(mode)
    if mode == 'train':
        mr_img_dir = 'trainMR'
        ct_img_dir = 'trainCT'
        img_list = os.path.join(dst_data_dir, 'train.txt')
    elif mode == 'test':
        mr_img_dir = 'testMR'
        ct_img_dir = 'testCT'
        img_list = os.path.join(dst_data_dir, 'test.txt')
    f = open(img_list, 'w')
    idx = 0
    for mr_f, ct_f in zip(mr_files, ct_files):
        mr_f, ct_f = os.path.join(src_data_dir, mr_f), os.path.join(src_data_dir, ct_f)
        mr_img = sitk.ReadImage(mr_f)
        ct_img = sitk.ReadImage(ct_f)
        mr_img = sitk.GetArrayFromImage(mr_img)
        ct_img = sitk.GetArrayFromImage(ct_img)
        mr_img = mr_img / 255
        ct_img = ct_img / 255
        pad_size = (np.asarray(input_patch_shape) - np.asarray(output_patch_shape)) // 2
        new_shape = np.asarray(mr_img.shape) + pad_size * 2
        new_shape = new_shape.astype('int')
        mr_img_padding = np.zeros(shape=(new_shape))
        # print(pad_size, new_shape, mr_img.shape)
        mr_img_padding[pad_size[0]:pad_size[0]+mr_img.shape[0], 
                        pad_size[1]:pad_size[1]+mr_img.shape[1], 
                        pad_size[2]:pad_size[2]+mr_img.shape[2]] = mr_img
        for mr_sub_img, ct_sub_img in zip(
            extract_ordered_overlap(mr_img_padding, input_patch_shape, stride_shape),
            extract_ordered_overlap(ct_img, output_patch_shape, stride_shape)
            ):
            mr_name = "{}/mr{}".format(mr_img_dir, idx)
            ct_name = "{}/ct{}".format(ct_img_dir, idx)
            f.write(mr_name + '\t' + ct_name + '\n')
            mr_name = os.path.join(dst_data_dir, mr_name)
            ct_name = os.path.join(dst_data_dir, ct_name)
            np.save(mr_name, mr_sub_img.astype('float32'))
            np.save(ct_name, ct_sub_img.astype('float32'))
            idx += 1

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int32) #spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled

def prepare_data_mr3t7t(src_data_dir, dst_data_dir, input_patch_shape=(32, 32, 32), output_patch_shape=(32, 32, 32),stride_shape=(16, 16, 16)):
    dirs = ['train3T', 'train7T', 'test3T', 'test7T']
    for d in dirs:
        path = os.path.join(dst_data_dir, d)
        if not os.path.exists(path):
            os.makedirs(path)
    src_dir3t = os.path.join(src_data_dir, '3T')
    src_dir7t = os.path.join(src_data_dir, '7T')
    file_paths_7t = os.listdir(src_dir7t)
    mr3t_files = []
    mr7t_files = []
    for f in os.listdir(src_dir3t):
        if f in file_paths_7t:
            file_path = os.path.join(src_dir3t, f)
            file_name = os.path.join(file_path, os.listdir(file_path)[0])
            file_path_7t = os.path.join(src_dir7t, f)
            file_name_7t = os.path.join(file_path_7t, os.listdir(file_path_7t)[0])
            mr3t_files.append(file_name)
            mr7t_files.append(file_name_7t)
            # print(file_name)
            # print(file_name_7t)
    # print(len(mr_file_pairs))
    train_mr3t_files, train_mr7t_files = mr3t_files[:-4], mr7t_files[:-4]
    test_mr3t_files, test_mr7t_files = mr3t_files[-4:], mr7t_files[-4:]
    _pre_data_mr3t7t(train_mr3t_files, train_mr7t_files, src_data_dir, dst_data_dir, input_patch_shape, output_patch_shape, stride_shape)
    _pre_data_mr3t7t(test_mr3t_files, test_mr7t_files, src_data_dir, dst_data_dir, input_patch_shape, output_patch_shape, stride_shape, mode='test')


def _pre_data_mr3t7t(mr_files, ct_files, src_data_dir, dst_data_dir, input_patch_shape, output_patch_shape, stride_shape, mode='train'):
    mode = mode.lower()
    assert mode in ['train', 'test', 'predict'], \
            "mode should be 'train' or 'test' or 'predict', but got {}".format(mode)
    if mode == 'train':
        mr_img_dir = 'train3T'
        ct_img_dir = 'train7T'
        img_list = os.path.join(dst_data_dir, 'train.txt')
    elif mode == 'test':
        mr_img_dir = 'test3T'
        ct_img_dir = 'test7T'
        img_list = os.path.join(dst_data_dir, 'test.txt')
    f = open(img_list, 'w')
    idx = 0
    for mr_f, ct_f in zip(mr_files, ct_files):
        # mr_f, ct_f = os.path.join(src_data_dir, mr_f), os.path.join(src_data_dir, ct_f)
        mr_img = sitk.ReadImage(mr_f)
        ct_img = sitk.ReadImage(ct_f)
        mr_img = resize_image_itk(mr_img, ct_img.GetSize())
        mr_img = sitk.GetArrayFromImage(mr_img)
        ct_img = sitk.GetArrayFromImage(ct_img)
        mr_img = mr_img / mr_img.max()
        ct_img = ct_img / ct_img.max()
        # pad_size = (np.asarray(input_patch_shape) - np.asarray(output_patch_shape)) // 2
        # new_shape = np.asarray(mr_img.shape) + pad_size * 2
        # new_shape = new_shape.astype('int')
        # mr_img_padding = np.zeros(shape=(new_shape))
        # # print(pad_size, new_shape, mr_img.shape)
        # mr_img_padding[pad_size[0]:pad_size[0]+mr_img.shape[0], 
        #                 pad_size[1]:pad_size[1]+mr_img.shape[1], 
        #                 pad_size[2]:pad_size[2]+mr_img.shape[2]] = mr_img
        for mr_sub_img, ct_sub_img in zip(
            extract_ordered_overlap(mr_img, input_patch_shape, stride_shape),
            extract_ordered_overlap(ct_img, output_patch_shape, stride_shape)
            ):
            mr_name = "{}/mr3t{}".format(mr_img_dir, idx)
            ct_name = "{}/mr7t{}".format(ct_img_dir, idx)
            f.write(mr_name + '\t' + ct_name + '\n')
            mr_name = os.path.join(dst_data_dir, mr_name)
            ct_name = os.path.join(dst_data_dir, ct_name)
            np.save(mr_name, mr_sub_img.astype('float32'))
            np.save(ct_name, ct_sub_img.astype('float32'))
            idx += 1


if __name__ == '__main__':
    # prepare_data('~/dealedMRCT')
    prepare_data_mr3t7t('/n01dat01/xlcheng/MR3T7T', '/n01dat01/xlcheng/datasetSRMR')
    # prepare_data_unet('/n01dat01/xlcheng/dealedMRCT', '/n01dat01/xlcheng/datasetGAN', input_patch_shape=(128, 128, 128), output_patch_shape=(128, 128, 128),stride_shape=(48, 48, 48))