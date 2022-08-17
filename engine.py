import sys
import math
import torch
import torch.nn as nn
from typing import Iterable, Dict

import utils.distributed_utils as utils


def train_one_epoch(model: Dict, criterion: nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, log_freq, max_norm: float = 0, freeze_embedder=False):
    
    for net in model.values():
        net.train()
    criterion.train()
    
    metric_logger = utils.LoaderwithLogger(delimiter="||")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.10f}'))
    
    if criterion.errors:
        for error in criterion.errors:
            metric_logger.add_meter(error, utils.SmoothedValue(window_size=1, fmt='{value:.10f}'))
    
    header = 'Epoch:[{:03d}]'.format(epoch)
    layer_out = [0, 1, 8, 11]
    for inputs in metric_logger.loadlog_every(data_loader, log_freq, header):
    
        for key, value in inputs.items():
            inputs[key] = value.to(device)
        
        l_inputs = model["embedder"](inputs["color_aug", 'l', 0],  layer_out=layer_out[:2])
        r_inputs = model["embedder"].forward_one(inputs["color_aug", 'r', 0])
        
        if freeze_embedder:
            cue_l = model["dcr"](l_inputs[-1].detach(), r_inputs.detach(), layer_out=layer_out[2:])
            outputs = model["refinenet"]([l_inputs[0].detach(), l_inputs[1].detach(), cue_l[0], cue_l[1]])
        else:
            cue_l = model["dcr"](l_inputs[-1], r_inputs, layer_out=layer_out[2:])
            outputs = model["refinenet"]([l_inputs[0], l_inputs[1], cue_l[0], cue_l[1]])
        
        loss_dict = criterion(inputs, outputs, model["dcr"])
    
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
        optimizer.step()
        
        with torch.no_grad():
            error_dict = criterion.compute_depth_errors(outputs[("depth", 0)], inputs["depth_gt"].squeeze(1))
        error_dict_reduced = utils.reduce_dict(error_dict)
        
        metric_logger.update(loss=loss_value, 
                             **loss_dict_reduced_scaled, 
                             **loss_dict_reduced_unscaled)
        metric_logger.update(**error_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return stats


@torch.no_grad()
def evaluate(model, criterion, data_loader, log_freq, device):
    
    for net in model.values():
        net.eval()
    criterion.eval()

    metric_logger = utils.LoaderwithLogger(delimiter="||")
    
    if criterion.errors:
        for error in criterion.errors:
            metric_logger.add_meter(error, utils.SmoothedValue(window_size=1, fmt='{value:.10f}'))
            
    header = 'Test:'
    layer_out = [0, 1, 8, 11]
    for inputs in metric_logger.loadlog_every(data_loader, log_freq, header):
        #during test, no DA
        for key, value in inputs.items():
            inputs[key] = value.to(device)
            
        l_inputs = model["embedder"](inputs["color", 'l', 0],  layer_out=layer_out[:2])
        r_inputs = model["embedder"].forward_one(inputs["color", 'r', 0])
        cue_l = model["dcr"](l_inputs[-1], r_inputs, layer_out=layer_out[2:])
        outputs = model["refinenet"]([l_inputs[0], l_inputs[1], cue_l[0], cue_l[1]])
        
        loss_dict = criterion(inputs, outputs)
        weight_dict = criterion.weight_dict
        error_dict = criterion.compute_depth_errors(outputs[("depth", 0)], inputs["depth_gt"].squeeze(1))

        # reduce losses and errors over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        
        error_dict_reduced = utils.reduce_dict(error_dict)
        
        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(**error_dict_reduced)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Averaged stats:", metric_logger)
    
    return stats
