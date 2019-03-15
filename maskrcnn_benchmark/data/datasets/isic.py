# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import os
import numpy as np

from torchvision import transforms as T

from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class ISICDataset():
    CLASSES = (
        "__background__ ",
        "cell",
    )

    def __init__(self, ann_file, root, mask_root, transforms=None):
        self.root = root
        self.mask_root = mask_root
        self.ids = os.listdir(root)

        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        self.annos, self.img_shapes = self.get_anno(ann_file)

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms
        self.basetransform = self.build_transform()

    @staticmethod
    def get_anno(anno_path):
        annos = {}
        shapes = {}
        with open(anno_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                name, bbox, shape = line.strip().split('\t')
                img_name = ISICDataset.mask_name2img_name(name)
                bbox = [float(x) for x in bbox.split(' ')]
                bbox.append(1)  # cls idx, annos is a list of size 5, x1 y1 x2 y2 cls
                shape = [int(x) for x in shape.split(' ')]
                annos[img_name] = bbox
                shapes[img_name] = shape
        return annos, shapes

    @staticmethod
    def img_name2mask_name(img_name):
        return img_name.replace(".jpg", "_segmentation.png")

    @staticmethod
    def mask_name2img_name(mask_name):
        return mask_name.replace("_segmentation.png", ".jpg")

    def __getitem__(self, idx):
        name = self.id_to_img_map[idx]
        mask_name = self.img_name2mask_name(name)
        img_path = os.path.join(self.root, name)
        mask_path = os.path.join(self.mask_root, mask_name)

        img = Image.open(img_path)
        mask = np.array(Image.open(mask_path))
        mask = mask / 255
        mask = mask.astype(np.uint8)

        # filter crowd annotations
        # TODO might be better to add an extra field
        annos = self.annos[name]
        boxes = [annos[:4]]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xyxy")

        classes = [annos[4]]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [mask]
        masks = SegmentationMask(masks, img.size, mode='mask')
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        name = self.id_to_img_map[index]
        img_data = self.img_shapes[name]
        return {"height": img_data[0], "width": img_data[1]}

    def get_gt_box(self, index):
        name = self.id_to_img_map[index]
        boxes = [self.annos[name][:4]]
        labels = [self.annos[name][4]]
        height, width = self.img_shapes[name]

        target = BoxList(boxes, (width, height), mode="xyxy")
        target.add_field("labels", torch.Tensor(labels))
        target.add_field("difficult", torch.tensor([False]))
        return target

    def get_image(self, index):
        name = self.id_to_img_map[index]
        img_path = os.path.join(self.root, name)

        img = Image.open(img_path).convert("RGB")
        size = img.size
        img = np.array(img)[:, :, [2, 1, 0]]
        # filter crowd annotations

        annos = self.annos[name]
        boxes = [annos[:4]]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, size, mode="xyxy")

        classes = [annos[4]]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img_tansformed = self.basetransform(img)

        return img, img_tansformed, target

    def get_gt_mask(self, index):
        name = self.id_to_img_map[index]
        mask_name = self.img_name2mask_name(name)
        mask_path = os.path.join(self.mask_root, mask_name)

        mask = np.array(Image.open(mask_path))
        mask = mask / 255
        mask = mask.astype(np.uint8)

        return mask

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        to_bgr_transform = T.Lambda(lambda x: x * 255)
        #TODO more general
        normalize_transform = T.Normalize(
            mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.]
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(600),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    @staticmethod
    def map_class_id_to_class_name(class_id):
        return ISICDataset.CLASSES[class_id]


if __name__ == '__main__':
    import numpy as np

    anno_file = '/data/lxc/Skin/ISIC-2017/detection_600/Training/annotations_ori.txt'
    root = '/data/mxj/Skin/ISIC-2017/images_600/Training'
    mask_root = '/data/mxj/Skin/ISIC-2017/annotations_600/Training'
    testDataset = ISICDataset(anno_file, root, mask_root)
    img, target, idx = testDataset[0]
    target_t = target.transpose(0)
    img_t = img.transpose(0)
    target_tt = target_t.transpose(0)

    mask = target.get_field('masks').masks[0].mask.numpy
    mask_tt = target.get_field('masks').masks[0].mask.numpy
    print(np.all(np.equal(mask, mask_tt)))
