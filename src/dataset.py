import os
import torch
import random
import rasterio
import numpy as np
from PIL import Image
from math import ceil, floor
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def hex_to_rgb(hex_code):
    return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))


def find_patch_overlaps(residual, num_patches, l_b, patch_size, overlap_info=None):
    """
    Input
    residual (list): list containing residual value, i.e. num_patches*patch_size - axis_dimension
    num_patches (list): number of patches
    l_b (int): the minimum possible overlap between adjacent patches

    returns overlap_info (list): each element informs on overlap between adjacent patches.
    """
    if overlap_info is None:
        overlap_info = []
    # base case
    if (num_patches[0] - 1) == 1:
        overlap_info.append(residual[0])
        return overlap_info
    for _ in range(num_patches[0] - 1):
        # u_b = residual[0] * patch_size - (num_patches[0] - 2) * ceil(residual[0] / num_patches[0]) # wrong u.b.
        u_b = residual[0] - (num_patches[0] - 2) * l_b
        valid_values = range(l_b, u_b + 1)
        # print(valid_values)
        el = random.choice(valid_values)
        overlap_info.append(el)
        num_patches[0] -= 1
        residual[0] -= el
        find_patch_overlaps(residual, num_patches, l_b, patch_size, overlap_info)
        break
    return overlap_info


def slicing(input, residual_val, num_patches, patch_size, overlap_info=None):
    """
    input: can either be original image_tensor or tuple of tensors.
    performs vertical or horizontal slicing
    """
    data = []
    # print(num_patches)
    lower_bound = int(ceil(residual_val / num_patches))
    if overlap_info is None:
        overlap_info = find_patch_overlaps([residual_val], [num_patches], lower_bound, patch_size)
    idx_start = 0
    idx_end = patch_size
    # is true when horizontal slicing has already been applied
    if isinstance(input, (tuple, list)):
        for i in input:
            for j in range(num_patches):
                if j == num_patches - 1:
                    patch = i[:, :, idx_start:]
                    data.append(patch)
                else:
                    patch = i[:, :, idx_start:idx_end]
                    data.append(patch)
                    idx_start += (patch_size - overlap_info[j])
                    idx_end += (patch_size - overlap_info[j])
            idx_start = 0
            idx_end = patch_size


    else:
        for j in range(num_patches):
            if j == num_patches - 1:
                patch = input[:, idx_start:, :]
                data.append(patch)
            else:
                patch = input[:, idx_start:idx_end, :]
                data.append(patch)
                idx_start += (patch_size - overlap_info[j])
                idx_end += (patch_size - overlap_info[j])

    return data, overlap_info


def get_patches(image_tensor, patch_size, overlapping_info=None):
    overlap_info = []
    if (image_tensor.shape[1] % patch_size) == 0 and (image_tensor.shape[2] % patch_size) == 0:
        # slice horizontally first -> thinking about maintaining some spatial info for when we re-patch?
        x = torch.split(image_tensor, patch_size, dim=1)
        data = [patch for i in x for patch in torch.split(i, patch_size, dim=2)]
    else:
        # data = []
        num_patches_x = int(ceil(image_tensor.shape[2] / patch_size))
        num_patches_y = int(ceil(image_tensor.shape[1] / patch_size))
        residual_x = (patch_size * num_patches_x) - image_tensor.shape[2]
        residual_y = (patch_size * num_patches_y) - image_tensor.shape[1]

        if (image_tensor.shape[1] % patch_size) == 0:
            x = torch.split(image_tensor, patch_size, dim=1)  # rename x, bad choice of variable name
            overlap_info_horizontal = None
            # now deal with vertical slicing
            data, overlap_info_vertical = slicing(x, residual_x, num_patches_x, patch_size, overlapping_info[1])

        else:
            # can't use torch.slicing directly
            if overlapping_info is None:
                x, overlap_info_horizontal = slicing(image_tensor, residual_y, num_patches_y, patch_size)
                data, overlap_info_vertical = slicing(x, residual_x, num_patches_x, patch_size)

            else:
                # print(overlapping_info[0], overlapping_info[1])
                x, overlap_info_horizontal = slicing(image_tensor, residual_y, num_patches_y, patch_size,
                                                     overlapping_info[0])
                data, overlap_info_vertical = slicing(x, residual_x, num_patches_x, patch_size, overlapping_info[1])

    # this condition is triggered only when patchifying labels
    if overlapping_info is not None:
        return data
    # patchifying images
    else:
        return data, overlap_info_horizontal, overlap_info_vertical


def get_mn_patches(input_tensor, number_patches, patch_size):
    patched_data = []
    #get stride in respective directions
    horizontal_stride = int(floor((input_tensor.shape[2] - patch_size) / number_patches[1]))
    vertical_stride = int(floor((input_tensor.shape[1] - patch_size) / number_patches[0]))
    idx_x_start = 0
    idx_x_end = patch_size
    idx_y_start = 0
    idx_y_end = patch_size
    for i in range(number_patches[0]):
        for j in range(number_patches[1]):
            patched_tensor = input_tensor[:, idx_y_start: idx_y_end, idx_x_start: idx_x_end]
            patched_data.append(patched_tensor)
            idx_x_start += horizontal_stride
            idx_x_end += horizontal_stride
        #reset horizontal indices and update vertical
        idx_x_start = 0
        idx_x_end = patch_size
        idx_y_start += vertical_stride
        idx_y_end += vertical_stride

    return patched_data


# pass path to label mask too?
class SegDataset(Dataset):
    all_classes = [('sand', 'ea7207'), ('soil', 'ffc000'), ('low_veg_bare', '96b810'), ('low_veg_non-bare', 'eeff00'),
                   ('high_veg_non-bare', '00ae7d'), ('high_veg_bare', '7eecdb'), ('urban_low', 'ffffff'),
                   ('urban_medium', 'fd00b5'), ('urban_high', 'ff0000'),
                   ('industry', '848484'), ('snow', '86d9ff'), ('water', '00007d'), ('rock', '584444')]


    def __init__(self, parent_pth, vert_split, patch_size=256, train=True, total_num_patches=None):
        """""
        parent_pth (string): path to parent directory. images and labels should be subdirectories. 
        img_pth (string): path image/s directory
        vert_split (boolean): specify if to split image/s vertically or horizontally
        patch_size (int): patch size n x n. Default is 256.
        patch_overlap (float): max overlap ratio
        train (boolean): train or test set. 
        classes (list): list of classes to extract from segmentation mask
        total_num_patches (int/iterable): if int then m x m patches, else m x n patches
        """""
        self.train = train
        self.root_dir = parent_pth
        # self.img_pth = img_pth
        self.vert_split = vert_split
        self.patch_size = patch_size
        self.total_num_patches = total_num_patches
        #self.classes = classes
        self.hex_rgb = {hex_to_rgb(self.all_classes[i][1]): i for i in range(len(self.all_classes))}
        #self.class_indices = [self.class_to_index[hex_to_rgb(colour)] for _, colour in self.all_classes if _ in self.classes]
        self.class_indices = [self.hex_rgb[hex_to_rgb(colour)] for _, colour in self.all_classes]
        self.sub_directories, self.image_files, self.label_files = self._get_files()

        if self.total_num_patches is None:
            self.images, overlap_information = self._get_data(self.sub_directories[0], self.image_files)
            self.labels = self._get_data(self.sub_directories[1], self.label_files,
                                         patch_overlap_info=overlap_information)
        else:

            if isinstance(self.total_num_patches, (tuple,list)):
                self.images = self._get_data(self.sub_directories[0], self.image_files,
                                             num_patches=self.total_num_patches)
                self.labels = self._get_data(self.sub_directories[1], self.label_files,
                                             num_patches=self.total_num_patches)
            else:
                self.total_num_patches = [self.total_num_patches] * 2
                self.images = self._get_data(self.sub_directories[0], self.image_files,
                                             num_patches=self.total_num_patches)
                self.labels = self._get_data(self.sub_directories[1], self.label_files,
                                             num_patches=self.total_num_patches)

    def _get_files(self):
        sub_directories = os.listdir(self.root_dir)
        sub_directories = sorted([sub_dir for sub_dir in sub_directories if
                                  os.path.isdir(os.path.join(self.root_dir, sub_dir)) and not sub_dir.startswith('.')])
        image_files = [file for file in os.listdir(os.path.join(self.root_dir, sub_directories[0])) if
                       os.path.isfile(os.path.join(self.root_dir, sub_directories[0], file))]
        label_files = [file for file in os.listdir(os.path.join(self.root_dir, sub_directories[1])) if
                       os.path.isfile(os.path.join(self.root_dir, sub_directories[1], file))]

        return sub_directories, image_files, label_files

    def _get_data(self, sub_directory, list_of_files, num_patches=None, patch_overlap_info=None):
        patched_data = []
        overlap_data = []
        # for file in self.image_files
        for f in range(len(list_of_files)):
            file = os.path.join(self.root_dir, sub_directory, list_of_files[f])
            # open image and convert to tensor
            if 'images' in sub_directory:
                with rasterio.open(file) as data:
                    img_data = data.read()
                # we only use this transform when transforming images, not masks
                # permute ensures correct ordering of dimensions, i.e. (H,W,C) is expected ordering of axes when applying ToTensor
                img_transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(lambda x: x.permute(1, 2, 0))])
                tensor_img = img_transform(img_data)
            else:
                with Image.open(file) as data:
                    tensor_img = torch.from_numpy(np.array(data)).to(torch.int)
                    tensor_img = torch.permute(tensor_img, (2, 0, 1))
                    tensor_img = tensor_img[:3, :, :]

            # split data into train/test and further split into patches
            if self.vert_split:
                dim = 2
                mid_point = tensor_img.shape[dim] // 2
            else:
                dim = 1
                mid_point = tensor_img.shape[dim] // 2
            if (tensor_img.shape[dim] % 2) != 0:
                if self.train:
                    data_tensors, _ = torch.split(tensor_img, [mid_point, mid_point + 1], dim=dim)
                else:
                    _, data_tensors = torch.split(tensor_img, [mid_point, mid_point + 1], dim=dim)
            else:
                if self.train:
                    data_tensors, _ = torch.split(tensor_img, mid_point, dim=dim)
                else:
                    _, data_tensors = torch.split(tensor_img, mid_point, dim=dim)
            if num_patches is None:
                # i.e. getting image patches
                if patch_overlap_info is None:
                    patches, overlap_horizontal, overlap_vertical = get_patches(data_tensors, self.patch_size)
                    overlap_data.append((overlap_horizontal, overlap_vertical))

                else:
                    # label patches
                    tensor_patches = get_patches(data_tensors, self.patch_size, patch_overlap_info[f])
                    patches = [self._rgb_to_label(tensor) for tensor in tensor_patches]
                    print(len(patches))
            else:
                # call function for get_mn_patches()
                if 'images' in sub_directory:
                    patches = get_mn_patches(data_tensors, num_patches, self.patch_size)
                else:
                    tensor_patches = get_mn_patches(data_tensors, num_patches, self.patch_size)
                    patches = [self._rgb_to_label(tensor) for tensor in tensor_patches]
            patched_data.extend(patches)

        if num_patches is None and patch_overlap_info is None:
            return patched_data, overlap_data
        else:
            return patched_data

    def _rgb_to_label(self, label_tensor):
        """
        Takes a label tensor (3, height, width) & converts to mask tensor (height, width) where each pixel corresponds
        to class index
        """
        mask_tensor = torch.zeros_like(label_tensor[0, :, :])

        for class_idx in self.class_indices:
            hex_colour = self.all_classes[class_idx][1]
            mask = torch.all(label_tensor == torch.tensor(hex_to_rgb(hex_colour)).view(3, 1, 1), dim=0)

            mask_tensor[mask] = class_idx
        return mask_tensor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# if __name__ == "__main__":
#     # dataset = SegDataset('../../../ste/rnd/User/yusuf/city_data/Berlin_Summer', vert_split=False, total_num_patches= [7, 8])
#     # #print(len(dataset))
#     # #print(len())
#     # # print(f' all classes: {dataset.all_classes}')
#     # #
#     # # print(f'classes to indices: {dataset.class_to_index}')
#     # # print(dataset.class_indices)
#     #
#     # for i, (image,label) in enumerate(dataset):
#     #     #print(label.shape)
#     #     #print(image.shape)
#     #     print(image.shape)
#     #     img_array = image.numpy()
#     #     print(img_array.shape)
#     #
#     #     #plt.imshow(np.transpose(img_array, (1,2,0)))
#     #     #plt.show()
#
#         # if i==0:
#         #     break
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     print(torch.cuda.is_available())
#
#     num_gpus = torch.cuda.device_count()
#
#     print(f'number of available gpus is: {num_gpus}')


