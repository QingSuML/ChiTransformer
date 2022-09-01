import os
import json
import random
import time
import datetime
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from dataset.kittidataset import *

import utils.distributed_utils as utils
from configs import TrainConfigs
from builder import build
from engine import train_one_epoch, evaluate

import wandb

def main(args):
    
    # scales: e.g. [0] or [0,1,2,3]
    args.num_scales = len(args.img_scales)
    
    utils.init_distributed_mode(args)

    log_folder = os.path.join(args.log_dir, args.model_name)
    output_dir = Path(args.output_dir)
    
    #default: "cuda"
    device = torch.device(args.device)

    #fixed seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # stereo training [0]
    assert args.frame_ids[0] == 0, "frame_ids must start with 0"
    
    if args.monocular:
        assert not args.stereo, "Stereo and monocular mode should be mutually exclusive."
        raise NotImplementedError

    # 0, master view, 's' reference view
    if args.stereo:
        assert args.frame_ids == [0], "Only one view is needed to find another view."
        args.frame_ids.append("s")
    
    args.num_dcr = args.depth - args.sa_depth
    
    chitransformer, criterion = build(args)
    
    chitransformer.to(device)
    criterion.to(device)
    
    if args.distributed:
        chitransformer = torch.nn.parallel.DistributedDataParallel(chitransformer,
                                                          device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = chitransformer.module
    
                
    models = {}
    parameters_to_train = []
    n_trainable_params = 0
    
    models['embedder'] = model_without_ddp.patch_embedder
    models["dcr"] = model_without_ddp.sa_dcr
    models['refinenet'] = model_without_ddp.refinenet
    
    if args.freeze_embedder:
        for params in models['embedder'].parameters():
            params.requires_grad_(False)
            
    else:
        parameters_to_train.append(
                {
                    "params":[params for params in models['embedder'].parameters() \
                              if params.requires_grad],
                    "lr": args.learning_rate_pretrained
                }
        )
        n_trainable_params += sum(p.numel() for p in parameters_to_train[-1]["params"])
    
    
    if not args.freeze_dcr_ca:
        parameters_to_train.append(
                    {
                        "params":[params for params in models['dcr'].DCR.parameters() \
                                if params.requires_grad],
                    }
        )
        n_trainable_params += sum(p.numel() for p in parameters_to_train[-1]["params"])
        
        parameters_to_train.append({"params": models['dcr'].pos_embed, "lr": args.learning_rate_pretrained})
        n_trainable_params += sum(p.numel() for p in parameters_to_train[-1]["params"])
    else:
        for params in models['dcr'].DCR.parameters():
            params.requires_grad_(False)
    
    if args.train_self_attention:
        parameters_to_train.append(
                {
                    "params":[params for name, params in models['dcr'].named_parameters() \
                             if "DCR" not in name and params.requires_grad and "pos_embed" not in name],
                    "lr": args.learning_rate_pretrained
                }
        )
        n_trainable_params += sum(p.numel() for p in parameters_to_train[-1]["params"])
    else:
        for params in models['dcr'].Blocks.parameters():
            params.requires_grad_(False)
    

    if not args.only_dcr and not args.freeze_dcr_ca:
        parameters_to_train.append(
                    {
                        "params":[params for _, params in models['refinenet'].head.named_parameters()\
                                if params.requires_grad],
                        "lr": args.learning_rate_pretrained
                    }
        )
        n_trainable_params += sum(p.numel() for p in parameters_to_train[-1]["params"])
    else:
        for params in models['refinenet'].head.parameters():
            params.requires_grad_(False)

    if args.train_refinenet:
        parameters_to_train.append(
                {
                    "params":[params for name, params in models['refinenet'].named_parameters() \
                             if "head" not in name and params.requires_grad],
                    "lr": args.learning_rate_pretrained
                }
        )
        n_trainable_params += sum(p.numel() for p in parameters_to_train[-1]["params"])
    else:
        for name, params in models['refinenet'].named_parameters():
            if "head" not in name:
                params.requires_grad_(False)
                        
    n_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f'number of total parameters: {n_params}\n')
    print(f'number of trainable parameters: {n_trainable_params}\n')


    """Load weights"""
    if args.load_weights:
        checkpoint = torch.load(args.load_weights, map_location='cpu')
        try:
            model_without_ddp.load_state_dict(checkpoint['model'])
        except:
            model_without_ddp.load_state_dict(checkpoint)

    """Optimizer"""
    optimizer = torch.optim.Adam(parameters_to_train, args.learning_rate,
                          weight_decay=args.weight_decay)
    """learning rate"""
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_drop, gamma=0.1)
    

    dataset_dict = {"kitti": KittiDataset,
                    "cityscapes": "CityScapes_Not_Implemented",
                    "argoverse": "Argoverse_Not_Implemented"}
    
    dataset = dataset_dict[args.dataset]
    
    if args.dataset == "kitti":
        fpath = os.path.join(os.path.dirname(__file__), "splits", args.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        num_train_samples = len(train_filenames)
    else:
        raise NotImplementedError(f"{args.dataset} is not implemented.")

    img_ext = '.png' if args.png else '.jpg'
    args.num_total_steps = (num_train_samples // (args.batch_size*args.world_size)) * args.epochs

    dataset_train = dataset(args.data_path, train_filenames, args.height, args.width,
                            args.frame_ids, args.num_scales, crop=args.crop, start_scale=0,
                            is_train=True, load_pred=args.pre_pred, img_ext=img_ext)
    dataset_val = dataset(args.data_path, val_filenames, args.height, args.width,
                          args.frame_ids, args.num_scales, crop=args.crop, start_scale=0,
                          is_train=False, img_ext=img_ext)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.batch_sampler:
        sampler_train = torch.util.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=sampler_train, num_workers=args.num_workers)
    else:
        data_loader_train = DataLoader(dataset_train, args.batch_size, sampler=sampler_train,
                                       num_workers=args.num_workers, drop_last=True)
    
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 num_workers=args.num_workers, drop_last=True)
        
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            
        model_without_ddp.load_state_dict(checkpoint['model'])
        
        if not args.eval and \
            'optimizer' in checkpoint and \
            'lr_scheduler' in checkpoint and \
            'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    
    if args.eval:
        test_stats = evaluate(models, criterion, data_loader_val, args.log_freq, device)
        if args.output_dir:
            utils.save_on_master(test_stats, output_dir / "eval.pth")
        return

    
    print("Training model named:\n  ", args.model_name)
    print("Models and WandB events files are saved to:\n  ", args.log_dir)
    print("Start training")
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            #In distributed mode, calling the set_epoch() method at the beginning of each epoch
            #before creating the DataLoader iterator is necessary to make shuffling work properly
            #across multiple epochs.
            sampler_train.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            models, criterion, data_loader_train, optimizer, device, epoch, args.log_freq,
            args.clip_max_norm, freeze_embedder=args.freeze_embedder)
        lr_scheduler.step()
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.check_period == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'training_configs': {**criterion.weight_dict, 'smoothness':args.smoothness_weight}
                }, checkpoint_path)

        test_stats = evaluate(models, criterion, data_loader_val, args.log_freq, device)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_params,
                     **criterion.weight_dict,
                     'smoothness': args.smoothness_weight}
        
        if utils.is_main_process():
            #wandb.log(log_stats)
            if args.output_dir:
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    
if __name__ == '__main__':
    
    #wandb.init(project="ChiTransformer Training")
    
    configs = TrainConfigs()
    args = configs.parse()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(args)
