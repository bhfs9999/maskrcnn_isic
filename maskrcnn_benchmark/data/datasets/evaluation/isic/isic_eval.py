import logging
import os
import torch
import numpy as np
from collections import OrderedDict, defaultdict
from tqdm import tqdm

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


def do_isic_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    eval_return = {}
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    if box_only:
        return

    logger.info("Preparing results for ISIC format")
    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        pred_boxlists, gt_boxlists, img_namelists = prepare_for_isic_detection(predictions, dataset)

        # save pred box coordiante
        save_detection_result(img_namelists, pred_boxlists, output_folder)

        logger.info("Evaluating bbox predictions")
        result = eval_detection_voc(
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            iou_thresh=0.5,
            use_07_metric=True,
        )
        result_str = "mAP: {:.4f}\n".format(result["map"])
        for i, ap in enumerate(result["ap"]):
            if i == 0:  # skip background
                continue
            result_str += "{:<16}: {:.4f}\n".format(
                dataset.map_class_id_to_class_name(i), ap
            )
        calc_overlap_rate(pred_boxlists, gt_boxlists)
        logger.info(result_str)
        eval_return["map"] = result["map"]

    if "segm" in iou_types:
        logger.info("Preparing segm results")
        segm_result = prepare_for_isic_segmentation(predictions, dataset)

        logger.info("Evaluating segmentation predictions")

        ious = np.zeros(len(segm_result))
        dices = np.zeros(len(segm_result))

        result_str = 'Segmentation Result:\n'

        for i, result in tqdm(enumerate(segm_result)):
            img_name = result["image_id"]
            pred_mask = result["segmentation"].squeeze().numpy()
            gt_mask = result["gt_mask"]
            score = result["score"]

            # Binarize masks
            gt = gt_mask > 0.5
            pr = pred_mask > 0.5

            ious[i] = IoU(gt, pr)
            dices[i] = Dice(gt, pr)

        result_str += "Mean IOU: {}".format(ious.mean())
        result_str += "Mean Dice: {}".format(dices.mean())

        eval_return["iou"] = ious.mean()
        eval_return["dice"] = dices.mean()

    logger.info(result_str)
    # check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w") as fid:
            fid.write(result_str)

    return eval_return

def save_detection_result(img_namelists, pred_boxlists, output_folder):
    with open(os.path.join(output_folder, "detection.txt"), "w") as f:
        for name, box in zip(img_namelists, pred_boxlists):
            box = box.bbox.numpy()[0].tolist()
            box = [str(round(x)) for x in box]
            f.write('{}\t{}\n'.format(name, ' '.join(box)))


def calc_overlap_rate(pred_boxlists, gt_boxlists, th=0.7):
    ious = []
    num = 0
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        iou = boxlist_iou(
            gt_boxlist,
            pred_boxlist,
        ).numpy()
        ious.append(iou)
        if iou > th:
            num += 1
    ratio = float(num) / len(ious)
    print("total num: %d, iou higher than %f: %d, ratio: %f" % (len(ious), th, num, ratio))
    return num, ratio


def get_highest_score_box(prediction):
    scores = prediction.get_field('scores')
    if len(scores) == 1:
        return prediction
    else:
        _, idx = scores.sort(0, descending=True)
        keep = idx[0].unsqueeze(0)
        best_bbox = prediction[keep]
        return  best_bbox


def prepare_for_isic_detection(predictions, dataset):
    # assert isinstance(dataset, ISISDataset)
    pred_boxlists = []
    gt_boxlists = []
    img_namelists = []
    for image_id, prediction in enumerate(predictions):
        if len(prediction) == 0:
            continue

        prediction = get_highest_score_box(prediction)
        img_info = dataset.get_img_info(image_id)

        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xyxy")
        pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_gt_box(image_id)
        gt_boxlists.append(gt_boxlist)

        img_name = dataset.get_img_name(image_id)
        img_namelists.append(img_name)

    return pred_boxlists, gt_boxlists, img_namelists


def prepare_for_isic_segmentation(predictions, dataset):
    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    isic_results = []
    for image_id, prediction in enumerate(predictions):
        if len(prediction) == 0:
            continue

        prediction = get_highest_score_box(prediction)
        original_id = dataset.id_to_img_map[image_id]
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        # t = time.time()
        # Masker is necessary only if masks haven't been already resized.
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0]
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        mapped_labels = labels

        gt_mask = dataset.get_gt_mask(image_id)
        isic_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": mask,
                    "score": scores[k],
                    "gt_mask": gt_mask,
                }
                for k, mask in enumerate(masks)
            ]
        )
    return isic_results


def eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec = calc_detection_voc_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap)}


def calc_detection_voc_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()

            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)


def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)
