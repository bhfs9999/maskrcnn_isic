import logging
import os
import torch
from collections import OrderedDict, defaultdict
import numpy as np
from tqdm import tqdm
import cv2

from pycocotools.cocoeval import COCOeval

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.data.datasets.cocowrapper import COCOWrapper


def do_cervix_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
    draw=False
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    # output_folder: '/data/lxc/output/maskrcnn/r50fpn_cervix_acid_bs4_lr.0005/inference/Cervix_acid_pos_test'
    dataset_name = output_folder.split('/')[-2]
    gt_cache_path = os.path.join(output_folder, '../../../../gt_cache/', dataset_name)

    if box_only:
        print("box only for cervix evaluation not support")
        return

    logger.info("Preparing results for cervix format")
    coco_gt = COCOWrapper(dataset, cache_path=gt_cache_path)
    coco_dt = COCOWrapper(dataset, predictions, cache_path=output_folder)
    results = COCOResults(*iou_types)

    recalls = {}
    box_num = 0

    for iou_type in iou_types:
        logger.info("Preparing {} results".format(iou_type))
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

        logger.info("Evaluating predictions")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        res = coco_eval
        results.update(res)
        recall = coco_eval.eval['recall'][0, :, 0, 2]
        box_num = np.mean([x.bbox.shape[0] for x in predictions])
        print("recall for each class: ", recall)
        print("avg predicted bbox: ", box_num)
        for i, r in enumerate(recall):
            recalls['{}_recall_{}'.format(iou_type, i)] = r

    logger.info(results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))

    if draw:
        draw_root = os.path.join(output_folder, 'vis')
        if not os.path.exists(draw_root):
            os.makedirs(draw_root)

        gts = defaultdict(list)
        dts = defaultdict(list)
        for gt in coco_gt.gt:
            gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in coco_dt.dt:
            dts[dt['image_id'], dt['category_id']].append(dt)

        visulization(dataset, gts, dts, draw_root)

    metrics = {
        'bbox_iou0.5': results.results['bbox']['AP50'],
        'segm_iou0.5': results.results['segm']['AP50'],
        'box_num': box_num
    }
    metrics.update(recalls)
    return metrics


def visulization(dataset, gts, dts, draw_root):
    gt_color = (255, 0, 0)
    pred_color = [(), (0, 255, 0), (0, 0, 255)]

    desc = "Visualizing result..."
    dt_template = "{}: {:.6f}"
    for image_id in tqdm(range(len(dataset)), desc=desc):
        image, _, _ = dataset.get_image(image_id)
        image = image[:, :, [2, 1, 0]]
        name = dataset.id_to_img_map[image_id]
        for cls_id in range(1, len(dataset.CLASSES)):
            gtboxes = gts[image_id, cls_id]
            dtboxes = dts[image_id, cls_id]
            for gtbox in gtboxes:
                # draw gt
                box = [int(x) for x in gtbox['bbox']]
                label = dataset.classid_to_name[int(gtbox['category_id'])]
                top_left, bottom_right = box[:2], box[2:]
                image = cv2.rectangle(
                    image, tuple(top_left), tuple(bottom_right), gt_color, 1
                )

                # write cls
                text_pos = (top_left[0], top_left[1] + 20)
                cv2.putText(
                    image, str(label), text_pos, cv2.FONT_HERSHEY_SIMPLEX, .4, gt_color, 1
                )
            for dtbox in dtboxes:
                box = [int(x) for x in dtbox['bbox']]
                label_id = int(gtbox['category_id'])
                label = dataset.classid_to_name[label_id]
                score = dtbox['score']
                top_left, bottom_right = box[:2], box[2:]

                # draw box
                image = cv2.rectangle(
                    image, tuple(top_left), tuple(bottom_right), pred_color[label_id], 1
                )

                # s = dt_template.format(label, score)
                #
                # # draw score
                # text_pos = (top_left[0], top_left[1] + 20)
                # cv2.putText(
                #     image, s, text_pos, cv2.FONT_HERSHEY_SIMPLEX, .4, pred_color[label_id], 1
                # )
        output_path = os.path.join(draw_root, name+'.jpg')
        cv2.imwrite(output_path, image)

# inspired from Detectron
def evaluate_box_proposals(
    predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("objectness").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)
