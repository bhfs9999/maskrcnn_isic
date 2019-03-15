# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import cv2
import torch
import torch.distributed as dist
from torchvision.utils import make_grid

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.structures.image_list import to_image_list


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    vis_period,
    arguments,
    cfg,
    tb_writer
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    vis_num = 0
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(v * cfg.SOLVER.LOSS_WEIGHT.MASK_WEIGHT if k == 'loss_mask'
                     else v * cfg.SOLVER.LOSS_WEIGHT.BOX_WEIGHT for k, v in loss_dict.items())
        # losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        loss_dict_reduced = {k:(v*cfg.SOLVER.LOSS_WEIGHT.MASK_WEIGHT if k=='loss_mask'
                                else v*cfg.SOLVER.LOSS_WEIGHT.BOX_WEIGHT) for k, v in loss_dict_reduced.items()}

        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        tb_writer.add_scalars('TrainLosses', loss_dict_reduced, global_step=iteration)
        tb_writer.add_scalar('TrainLoss', losses_reduced, global_step=iteration)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % vis_period == 0:
            # visualize predict box
            # set model to eval mode
            model.eval()
            vis_image, vis_image_transformed, target = data_loader.dataset.get_image(vis_num)
            image_list = to_image_list(vis_image_transformed, cfg.DATALOADER.SIZE_DIVISIBILITY)
            image_list = image_list.to(device)
            cpu_device = torch.device("cpu")
            with torch.no_grad():
                predictions = model(image_list)
                predictions = [o.to(cpu_device) for o in predictions]

            # only one picture
            predictions = predictions[0]
            top_predictions = select_topn_predictions(predictions, 3)

            # visualize
            result = vis_image.copy()
            result = overlay_boxes_cls_names(result, top_predictions, target)

            result = torch.from_numpy(result)
            result = result.permute(2, 0, 1)[None, :, :, :]
            result = make_grid([result])
            tb_writer.add_image('Image_train', result, iteration)
            model.train()
            vis_num += 1
            vis_num %= len(data_loader.dataset)
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def select_topn_predictions(predictions, topn=3):
    """
    Select only predictions which have a `score` > th,
    and returns the top n predictions in descending order of score

    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.
        topn (int): how mant box to keep
        th (float): score threshold

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    topn = min(topn, len(scores))
    _, idx = scores.sort(0, descending=True)
    idx = idx[:topn]
    return predictions[idx]


def overlay_boxes_cls_names(image, predictions, targets):
    """
    Adds the predicted boxes with score on top of the image with
    Adds the gt boxes

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    gt_color = (255, 0, 0)
    pred_color = (0, 255, 0)

    # draw gt
    gtbox = targets.bbox[0].long()
    top_left, bottom_right = gtbox[:2].tolist(), gtbox[2:].tolist()
    image = cv2.rectangle(
        image, tuple(top_left), tuple(bottom_right), gt_color, 2
    )

    # draw prediction
    boxes = predictions.bbox
    scores = predictions.get_field("scores").tolist()

    template = "{:.2f}"
    for box, score in zip(boxes, scores):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()

        # draw box
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), pred_color, 2
        )
        s = template.format(score)

        # draw score
        text_pos = (top_left[0], top_left[1] + 20)
        cv2.putText(
            image, s, text_pos, cv2.FONT_HERSHEY_SIMPLEX, .8, pred_color, 1
        )

    return image
