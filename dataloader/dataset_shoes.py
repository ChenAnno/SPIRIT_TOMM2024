import PIL
import PIL.Image
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import os
import json as jsonmod


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim=288):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio=1.25, dim=288):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class ShoesDataset(Dataset):
    def __init__(self, split, mode="relative", preprocess=targetpad_transform(target_ratio=1.25, dim=640)):
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.shoes_path = "/mnt/vision_retrieval/chenyanzhe/shoes_dataset/"
        self.local_feature_path = '/mnt/vision_retrieval/chenyanzhe/shoes_dataset/shoes_local_feature_13/'
        self.image_id2name = self.load_file(os.path.join(self.shoes_path, f'split.{split}.json'))
        if self.mode == "relative":
            self.annotations = self.load_file(os.path.join(self.shoes_path, f'triplet.{split}.json'))

    def __getitem__(self, index):
        if self.mode == "relative":  # 返回三元组形式
            ann = self.annotations[index]
            caption = ann['RelativeCaption']
            reference_path = self.shoes_path + ann['ReferenceImageName']
            target_path = self.shoes_path + ann['ImageName']
            reference_name = reference_path.split('/')[-1].split(".jpg")[0]
            target_name = target_path.split('/')[-1].split(".jpg")[0]

            ref_local_path = self.local_feature_path + f"{reference_name}.pth"
            ref_patch_feat = torch.load(ref_local_path)
            tar_local_path = self.local_feature_path + f"{target_name}.pth"
            tar_patch_feat = torch.load(tar_local_path)

            if self.split == "train":
                reference_image = self.preprocess(PIL.Image.open(reference_path))
                target_image = self.preprocess(PIL.Image.open(target_path))
                return reference_image, target_image, caption, ref_patch_feat, tar_patch_feat
            else:  # val
                return reference_name, target_name, caption, ref_patch_feat, tar_patch_feat
        else:
            image_path = self.shoes_path + self.image_id2name[index]
            image_name = image_path.split('/')[-1].split(".jpg")[0]
            image = self.preprocess(PIL.Image.open(image_path))

            local_patch_path = self.local_feature_path + image_name.split(".jpg")[0] + ".pth"
            local_feature = torch.load(local_patch_path)

            return image_name, image, local_feature

    def __len__(self):
        if self.mode == "relative":
            return len(self.annotations)
        else:
            return len(self.image_id2name)

    def load_file(self, f):
        with open(f, "r") as jsonfile:
            ann = jsonmod.loads(jsonfile.read())
        return ann


if __name__ == '__main__':
    shoes_dataset = ShoesDataset(split="val", mode="relative")
    print(shoes_dataset[0])

