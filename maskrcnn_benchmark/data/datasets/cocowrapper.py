from tqdm import tqdm
import numpy as np
import os
import pickle as pkl

from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
import pycocotools.mask as mask_utils
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker


class COCOWrapper(object):
    """
    Mimics ONLY the basic utilities from pycocotools.coco.COCO class which are
    required for and used by pycocotools.coco.COCOeval
    The implementation focuses to cover the bare minimum to make the script
    running.
    """

    def __init__(self, dataset, every_prediction=None, cache_path=None):
        # follow COCO notation: gt -> ground truth, dt -> detection
        # COCO API requires data to be held in the memory throughout the eval
        # so fingers crossed that segmentation masks converted to RLE can fit in

        self.dataset = dataset
        self.every_prediction = every_prediction
        self.cache_path = cache_path

        if every_prediction is None:
            # This COCOWrapper instance will hold only GT annotations
            self.gt = self._buildCOCOAnnotations()
            self.dt = None
        else:
            # This COCOWrapper instance will hold only predictions
            self.gt = None
            self.dt = self._buildCOCOPredictions()

    def getAnnIds(self, *args, **kwargs):
        # AnnIds is not a necessary thing
        return

    def getCatIds(self, *args, **kwargs):
        return list(self.dataset.classid_to_name)

    def getImgIds(self, *args, **kwargs):
        # ImgIds is not a necessary thing, so just send back a list from [0..N]
        return list(range(len(self.dataset)))

    def _buildCOCOAnnotations(self):
        gt_cache_name = 'len_{}.pkl'.format(len(self.dataset))
        gt_cache_path = os.path.join(self.cache_path, gt_cache_name)
        if os.path.exists(gt_cache_path):
            print("Loading COCO GT annots from: ", gt_cache_path)
            coco_anns = pkl.load(open(gt_cache_path, 'rb'))
        else:
            print("Building COCO GT annots", flush=True)
            desc = "Parsing images"
            coco_anns = []
            for image_id in tqdm(range(len(self.dataset)), desc=desc):
                _, anns, _ = self.dataset[image_id]
                for inst_idx in range(len(anns)):
                    # TODO: find out why BoxList indexing would be a problem.
                    # Only ranges can be applied ATM.
                    ann = anns[inst_idx:inst_idx + 1]
                    ann = {
                        "id": 1+inst_idx,
                        "image_id": image_id,
                        "size": ann.size,
                        "bbox": ann.bbox[0].tolist(),
                        "area": ann.area().item(),
                        "category_id": ann.get_field("labels").item(),
                        "segmentation": ann.get_field("masks"),
                        "iscrowd": 0
                    }
                    ann["segmentation"] = self.annToRLE(ann)
                    coco_anns.append(ann)
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
            print("Dumping COCO GT annots", flush=True)
            pkl.dump(coco_anns, open(gt_cache_path, 'wb'))
        return coco_anns

    def _buildCOCOPredictions(self):
        dt_cache_name = 'len_{}.pkl'.format(len(self.dataset))
        dt_cache_path = os.path.join(self.cache_path, dt_cache_name)
        if os.path.exists(dt_cache_path):
            print("Loading COCO DT annos from: ", dt_cache_path)
            coco_preds = pkl.load(open(dt_cache_path, 'rb'))
        else:
            print("Building COCO Predictions", flush=True)
            desc = "Parsing images"
            masker = Masker(threshold=0.5, padding=1)
            coco_preds = []
            for image_id, predictions in tqdm(enumerate(self.every_prediction), desc=desc):
                if len(predictions) == 0:
                    continue

                img_info = self.dataset.get_img_info(image_id)
                width = img_info["width"]
                height = img_info["height"]
                if predictions.size[0] != width or predictions.size[1] != height:
                    predictions = predictions.resize(size=(width, height))

                for inst_idx in range(len(predictions)):
                    pred = predictions[inst_idx:inst_idx + 1]
                    mask = pred.get_field('mask')
                    if list(mask.shape[-2:]) != [height, width]:
                        mask = masker(mask.expand(1, -1, -1, -1, -1), pred)
                        mask = mask[0].squeeze()
                    segm = SegmentationMask([mask], pred.size, mode='mask')
                    pred = {
                        "id": 1+inst_idx,
                        "image_id": image_id,
                        "size": pred.size,
                        "bbox": pred.bbox[0].tolist(),
                        "area": pred.area().item(),
                        "segmentation": segm,
                        "category_id": pred.get_field("labels").item(),
                        "score": pred.get_field("scores").item(),  # preds differ here
                        "iscrowd": 0
                    }
                    pred["segmentation"] = self.annToRLE(pred)
                    coco_preds.append(pred)
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
            print("Dumping COCO DT annots", flush=True)
            pkl.dump(coco_preds, open(dt_cache_path, 'wb'))
        return coco_preds

    def loadAnns(self, *args, **kwargs):

        if self.dt is None:
            return self.gt
        else:
            return self.dt

    def annToRLE(self, ann):
        segm = ann['segmentation']
        h, w = ann['size']

        if isinstance(segm, dict) and "counts" in segm.keys():
            # already rle
            rle = segm
        elif isinstance(segm, SegmentationMask) and segm.mode == 'poly':
            segm = ann.instances.polygons
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(segm, h, w)
            rle = mask_utils.merge(rles)
        elif isinstance(segm, SegmentationMask) and segm.mode == 'mask':
            np_mask = np.array(segm.masks[0].mask, order="F").astype(np.uint8)
            rle = mask_utils.encode(np_mask)
        else:
            raise RuntimeError("Unknown segmentation format: %s" % segm)

        return rle