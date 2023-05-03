from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np

class ImgDataset(Dataset):
    def __init__(self, file_list, data_dir):
        self.data_dir = data_dir
        self.train_images = []
        self.label_images = []
        list_file = os.path.join(self.data_dir, file_list)
        with open(list_file, 'r') as f:
            for line in f.readlines():
                image, label = line.strip().split('\t')
                self.train_images.append(image)
                self.label_images.append(label)

    def _load_img(self, path):
        path = os.path.join(self.data_dir, '{}.npy'.format(path))
        return np.load(path)

    def __getitem__(self, idx):
        train_image = self._load_img(self.train_images[idx])
        label_image = self._load_img(self.label_images[idx])
        train_image = train_image[np.newaxis, :].astype('float32')
        label_image = label_image[np.newaxis, :].astype('float32')
        # sample = {"image": train_image, "label": label_image}
        return train_image, label_image
    
    def __len__(self):
        return len(self.train_images)


if __name__ == '__main__':
    training_data = ImgDataset('train.txt', 'dataset')
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)