import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import image


class RGBDataset(Dataset):
    def __init__(self, dataset_dir, has_gt):
        """
        In:
            img_dir: string, path of train, val or test folder.
            has_gt: bool, indicating if the dataset has ground truth masks.
        Out:
            None.
        Purpose:
            Initialize instance variables.
        """
        # Input normalization info
        mean_rgb = [0.722, 0.751, 0.807]
        std_rgb = [0.171, 0.179, 0.197]

        self.dataset_dir = dataset_dir
        self.has_gt = has_gt
        # transform to be applied on a sample.
        # For this homework, compose ToTensor() and normalization for RGB image should be enough.
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=mean_rgb, std=std_rgb),
             ])
        # number of samples in the dataset.
        # You'd better not hard code the number, because this class is used to create train, validation and test dataset.
        filenames = os.listdir(os.path.join(self.dataset_dir, 'rgb'))
        self.dataset_length = len(filenames)
        if '.DS_Store' in filenames:
            self.dataset_length -= 1

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        """
        In:
            idx: int, index of each sample, in range(0, dataset_length).
        Out:
            sample: a dictionary that stores paired rgb image and corresponding ground truth mask (if available).
                    rgb_img: Tensor [3, height, width]
                    target: Tensor [height, width], use torch.LongTensor() to convert.
        Purpose:
            Given an index, return paired rgb image and ground truth mask as a sample.
        Hint:
            Use image.read_rgb() and image.read_mask() to read the images.
            Look at the filenames and think about how to associate idx with the file name of images.
            Remember to apply transform on the sample.
        """
        path_to_rgb = os.path.join(self.dataset_dir, 'rgb', f'{idx}_rgb.png')
        rgb_img = image.read_rgb(path_to_rgb)
        rgb_img = self.transform(rgb_img) # need to convert to tensor because CNN can only operate on tensors not arrays
        if self.has_gt is False:
            sample = {'input': rgb_img}
        else:
            path_to_mask = os.path.join(self.dataset_dir, 'gt', f'{idx}_gt.png')
            gt_mask = image.read_mask(path_to_mask)
            gt_mask = torch.LongTensor(gt_mask) # need to convert to tensor because CNN can only operate on tensors not arrays
            sample = {'input': rgb_img, 'target': gt_mask}
        return sample

if __name__ == "__main__":
    from segmentation_helper import check_dataset
    dataset = RGBDataset("dataset/train", has_gt=True)
    check_dataset(dataset)