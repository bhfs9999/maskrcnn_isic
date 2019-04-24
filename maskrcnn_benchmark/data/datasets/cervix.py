# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import os
import pickle as pkl

from torchvision import transforms as T

from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class CervixDataset:
    CLASSES = (
        "__background__ ",
        "lsil",
        "hsil",
    )

    classid_to_name = {
        # 0: "unlabeled"  -  exclude background category, otherwise it will
        #                   influence the mAP score
        1: "lsil",
        2: "hsil"
    }

    label_map = [0, 1, 2, 0]

    def __init__(self, ann_file, img_root, mask_root, split_file, transforms=None):
        self.img_root = img_root
        self.mask_root = mask_root
        with open(split_file) as f:
            self.ids = [x.strip() for x in f.readlines()]
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        self.annos = self.get_anno(ann_file)

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms
        self.basetransform = self.build_transform()

    def get_anno(self, anno_path):
        annos = pkl.load(open(anno_path, 'rb'))
        for k, v in annos.items():
            result = []
            for anno in v['annos']:
                anno['label'] = self.label_map[anno['label']]
                if anno['label'] != 0:
                    result.append(anno)
            v['annos'] = result
        return annos

    @staticmethod
    def name2mask_name(name):
        return name + ".gif"

    @staticmethod
    def name2img_name(name):
        return name + ".jpg"

    def __getitem__(self, idx):
        name = self.id_to_img_map[idx]
        img_name = self.name2img_name(name)
        img_path = os.path.join(self.img_root, img_name)

        img = Image.open(img_path)

        # TODO might be better to add an extra field
        annos = self.annos[name]['annos']
        boxes = [anno['box'] for anno in annos]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xyxy")

        labels = [anno['label'] for anno in annos]
        labels = torch.tensor(labels)
        if not (labels<3).all():
            print(name)
        target.add_field("labels", labels)

        masks = [anno['segmentation'] for anno in annos]
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
        img_data = self.annos[name]['shape']
        return {"height": img_data[0], "width": img_data[1]}


    def get_img_name(self, index):
        return self.id_to_img_map[index]


    def get_gt_box(self, index):
        name = self.id_to_img_map[index]
        annos = self.annos[name]['annos']
        boxes = [anno['box'] for anno in annos]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        labels = [anno['label'] for anno in annos]
        height, width = self.annos[name]['shape']

        target = BoxList(boxes, (width, height), mode="xyxy")
        target.add_field("labels", torch.Tensor(labels))
        target.add_field("difficult", torch.tensor([False]))
        return target

    def get_image(self, index):
        name = self.id_to_img_map[index]
        img_name = self.name2img_name(name)
        img_path = os.path.join(self.img_root, img_name)

        img = Image.open(img_path).convert("RGB")
        size = img.size
        img = np.array(img)

        annos = self.annos[name]['annos']
        boxes = [anno['box'] for anno in annos]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, size, mode="xyxy")

        labels = [anno['label'] for anno in annos]
        labels = torch.tensor(labels)
        target.add_field("labels", labels)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            # convert rgb to bgr
            img_tansformed = self.basetransform(img)
        else:
            img_tansformed = img

        return img, img_tansformed, target

    @staticmethod
    def build_transform():
        """
        Creates a basic transformation that was used to train the models
        """
        # the input image is rgb, first we need to convert to bgr
        # then what we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]] * 255)
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
        return CervixDataset.CLASSES[class_id]


if __name__ == '__main__':
    import numpy as np

    anno_file = '/data/lxc/Cervix/detection/annos/anno.pkl'
    img_root = '/data/mxj/Cervix/Segmentation/cervix_resize_600_segmentation/Images'
    mask_root = '/data/mxj/Cervix/Segmentation/cervix_resize_600_segmentation/Masks'
    split_file = '/data/mxj/Cervix/Segmentation/cervix_resize_600_segmentation/data_split/acid/train_pos.txt'
    testDataset = CervixDataset(anno_file, img_root, mask_root, split_file)
    img, target, idx = testDataset[0]
    target_t = target.transpose(0)
    img_t = img.transpose(0)
    target_tt = target_t.transpose(0)

    mask = target.get_field('masks').masks[0].mask.numpy
    mask_tt = target.get_field('masks').masks[0].mask.numpy
    print(np.all(np.equal(mask, mask_tt)))
