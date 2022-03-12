from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch
# import cv2
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from tqdm import tqdm

logger = logging.getLogger(__name__)

# scaler = GradScaler()


def train(config, train_loader, model, criterion, criterion_kd, consistency_weight,
          kd_weight, ens_weight, optimizer, epoch, output_dir, tb_log_dir, writer_dict):

    model.train()

    # end = time.time()
    logger.info(f'Current Epoch: {epoch}')

    with tqdm(total=len(train_loader)) as t:

        for i, (input, target, target_weight, meta) in enumerate(train_loader):

            optimizer.zero_grad()

            # measure data loading time
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            # with autocast():
            outputs, ens_outputs = model(input)

            loss_hard = 0
            loss_soft = 0
            for output in outputs[0:]:
                loss_hard += criterion(output, target, target_weight)
                loss_soft += criterion_kd(output, ens_outputs, target_weight)

            loss_ens = ens_weight * criterion(ens_outputs, target, target_weight)
            loss_soft = kd_weight * consistency_weight * loss_soft
            # print(loss_hard)
            loss = loss_hard + loss_ens + loss_soft

            # compute gradient and do update step
            loss.backward()
            # scaler.scale(loss).backward()
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()

            t.update()
            t.set_postfix(loss_ens=f'{loss_ens:5f}', loss_hard=f'{loss_hard:.5f}', loss_soft=f'{loss_soft:.7f}')

        t.close()


def get_current_consistency_weight(current, rampup_length):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def validate(config, val_loader, val_dataset, model, criterion, epoch, output_dir,
             tb_log_dir, writer_dict=None):

    acc_ens = AverageMeter()
    acc_b1 = AverageMeter()
    acc_b2 = AverageMeter()
    acc_b3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds_ens = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32
    )

    all_preds_b1 = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32
    )

    all_preds_b2 = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32
    )

    all_preds_b3 = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32
    )

    all_boxes_ens = np.zeros((num_samples, 6))
    all_boxes_b1 = np.zeros((num_samples, 6))
    all_boxes_b2 = np.zeros((num_samples, 6))
    all_boxes_b3 = np.zeros((num_samples, 6))

    image_path = []
    filenames = []
    imgnums = []
    idx_ens = 0
    idx_b1 = 0
    idx_b2 = 0
    idx_b3 = 0

    with torch.no_grad():
        # end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            input = input.cuda(non_blocking=True)
            outputs, ens_outputs = model(input)

            output_ens = ens_outputs

            # stack=2
            # output_b1 = outputs[1]
            # output_b2 = outputs[3]
            # output_b3 = outputs[5]

            # stack=4
            output_b1 = outputs[-3]
            output_b2 = outputs[-2]
            output_b3 = outputs[-1]

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped, ens_flipped = model(input_flipped)

                output_flipped_ens = ens_flipped

                output_flipped_b1 = outputs_flipped[-3]
                output_flipped_b2 = outputs_flipped[-2]
                output_flipped_b3 = outputs_flipped[-1]

                # -----------------------------------Ensemble Result
                output_flipped_ens = flip_back(output_flipped_ens.cpu().numpy(),
                                               val_dataset.flip_pairs)
                output_flipped_ens = torch.from_numpy(output_flipped_ens.copy()).cuda()
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped_ens[:, :, :, 1:] = \
                        output_flipped_ens.clone()[:, :, :, 0:-1]
                output_ens = (output_ens + output_flipped_ens) * 0.5

                # ------------------------------------Branch 1
                output_flipped_b1 = flip_back(output_flipped_b1.cpu().numpy(),
                                              val_dataset.flip_pairs)
                output_flipped_b1 = torch.from_numpy(output_flipped_b1.copy()).cuda()
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped_b1[:, :, :, 1:] = \
                        output_flipped_b1.clone()[:, :, :, 0:-1]
                output_b1 = (output_b1 + output_flipped_b1) * 0.5

                # -------------------------------------Branch 2
                output_flipped_b2 = flip_back(output_flipped_b2.cpu().numpy(),
                                              val_dataset.flip_pairs)
                output_flipped_b2 = torch.from_numpy(output_flipped_b2.copy()).cuda()
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped_b2[:, :, :, 1:] = \
                        output_flipped_b2.clone()[:, :, :, 0:-1]
                output_b2 = (output_b2 + output_flipped_b2) * 0.5

                # --------------------------------------Branch 3
                output_flipped_b3 = flip_back(output_flipped_b3.cpu().numpy(),
                                              val_dataset.flip_pairs)
                output_flipped_b3 = torch.from_numpy(output_flipped_b3.copy()).cuda()
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped_b3[:, :, :, 1:] = \
                        output_flipped_b3.clone()[:, :, :, 0:-1]
                output_b3 = (output_b3 + output_flipped_b3) * 0.5

            target = target.cuda(non_blocking=True)
            num_images = input.size(0)

            # ------------------Branch Ensemble
            _, avg_acc_ens, cnt, _ = accuracy(output_ens.cpu().numpy(),
                                              target.cpu().numpy())
            acc_ens.update(avg_acc_ens, cnt)

            # ------------------Branch 1
            _, avg_acc_b1, cnt, _ = accuracy(output_b1.cpu().numpy(),
                                             target.cpu().numpy())
            acc_b1.update(avg_acc_b1, cnt)

            # ------------------Branch 2
            _, avg_acc_b2, cnt, _ = accuracy(output_b2.cpu().numpy(),
                                             target.cpu().numpy())
            acc_b2.update(avg_acc_b2, cnt)

            # ------------------Branch 3
            _, avg_acc_b3, cnt, _ = accuracy(output_b3.cpu().numpy(),
                                             target.cpu().numpy())
            acc_b3.update(avg_acc_b3, cnt)

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            # ------------------Branch Ensemble
            preds_ens, maxvals_ens = get_final_preds(
                config, output_ens.cpu().numpy(), c, s)

            # ------------------Branch 1
            preds_b1, maxvals_b1 = get_final_preds(
                config, output_b1.cpu().numpy(), c, s)

            # ------------------Branch 2
            preds_b2, maxvals_b2 = get_final_preds(
                config, output_b2.cpu().numpy(), c, s)

            # ------------------Branch 3
            preds_b3, maxvals_b3 = get_final_preds(
                config, output_b3.cpu().numpy(), c, s)

            image_path.extend(meta['image'])

            # ------------------Branch Ensemble
            all_preds_ens[idx_ens:idx_ens + num_images, :, 0:2] = preds_ens[:, :, 0:2]
            all_preds_ens[idx_ens:idx_ens + num_images, :, 2:3] = maxvals_ens
            all_boxes_ens[idx_ens:idx_ens + num_images, 0:2] = c[:, 0:2]
            all_boxes_ens[idx_ens:idx_ens + num_images, 2:4] = s[:, 0:2]
            all_boxes_ens[idx_ens:idx_ens + num_images, 4] = np.prod(s * 200, 1)
            all_boxes_ens[idx_ens:idx_ens + num_images, 5] = score
            idx_ens += num_images

            # ------------------Branch 1
            all_preds_b1[idx_b1:idx_b1 + num_images, :, 0:2] = preds_b1[:, :, 0:2]
            all_preds_b1[idx_b1:idx_b1 + num_images, :, 2:3] = maxvals_b1
            all_boxes_b1[idx_b1:idx_b1 + num_images, 0:2] = c[:, 0:2]
            all_boxes_b1[idx_b1:idx_b1 + num_images, 2:4] = s[:, 0:2]
            all_boxes_b1[idx_b1:idx_b1 + num_images, 4] = np.prod(s * 200, 1)
            all_boxes_b1[idx_b1:idx_b1 + num_images, 5] = score
            idx_b1 += num_images

            # ------------------Branch 2
            all_preds_b2[idx_b2:idx_b2 + num_images, :, 0:2] = preds_b2[:, :, 0:2]
            all_preds_b2[idx_b2:idx_b2 + num_images, :, 2:3] = maxvals_b2
            all_boxes_b2[idx_b2:idx_b2 + num_images, 0:2] = c[:, 0:2]
            all_boxes_b2[idx_b2:idx_b2 + num_images, 2:4] = s[:, 0:2]
            all_boxes_b2[idx_b2:idx_b2 + num_images, 4] = np.prod(s * 200, 1)
            all_boxes_b2[idx_b2:idx_b2 + num_images, 5] = score
            idx_b2 += num_images

            # ------------------Branch 3
            all_preds_b3[idx_b3:idx_b3 + num_images, :, 0:2] = preds_b3[:, :, 0:2]
            all_preds_b3[idx_b3:idx_b3 + num_images, :, 2:3] = maxvals_b3
            all_boxes_b3[idx_b3:idx_b3 + num_images, 0:2] = c[:, 0:2]
            all_boxes_b3[idx_b3:idx_b3 + num_images, 2:4] = s[:, 0:2]
            all_boxes_b3[idx_b3:idx_b3 + num_images, 4] = np.prod(s * 200, 1)
            all_boxes_b3[idx_b3:idx_b3 + num_images, 5] = score
            idx_b3 += num_images

        # ------------------Branch Ensemble
        name_values_ens, perf_indicator_ens = val_dataset.evaluate(
            config, all_preds_ens, output_dir, all_boxes_ens, image_path, filenames, imgnums
        )
        # ------------------Branch 1
        name_values_b1, perf_indicator_b1 = val_dataset.evaluate(
            config, all_preds_b1, output_dir, all_boxes_b1, image_path, filenames, imgnums
        )
        # ------------------Branch 2
        name_values_b2, perf_indicator_b2 = val_dataset.evaluate(
            config, all_preds_b2, output_dir, all_boxes_b2, image_path, filenames, imgnums
        )
        # ------------------Branch 3
        name_values_b3, perf_indicator_b3 = val_dataset.evaluate(
            config, all_preds_b3, output_dir, all_boxes_b3, image_path, filenames, imgnums
        )

        model_name = config.MODEL.NAME

        # ------------------Branch Ensemble
        if isinstance(name_values_ens, list):
            for name_value in name_values_ens:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values_ens, model_name)

        # ------------------Branch 1
        if isinstance(name_values_b1, list):
            for name_value in name_values_b1:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values_b1, model_name)

        # ------------------Branch 2
        if isinstance(name_values_b2, list):
            for name_value in name_values_b2:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values_b2, model_name)

        # ------------------Branch 3
        if isinstance(name_values_b3, list):
            for name_value in name_values_b3:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values_b3, model_name)


        score = np.array([perf_indicator_b1, perf_indicator_b2, perf_indicator_b3])
        max_score = np.max(score)
        max_branch = np.argmax(score)

    return max_score, (max_branch + 1)


def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
