import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class TrainConfigs(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="ChiTransformer Training Configurations")
        
        # model configurations:
        self.parser.add_argument("--crop", help="crop the input image", action='store_true')
        self.parser.add_argument("--dcr_mode", type=str, help="depth cue rectification mode", default="direct")
        self.parser.add_argument("--depth", type=int, help="depth of attention layer", default=12)
        self.parser.add_argument("--embed_dim", type=int, help="layers of embedder", default=768)
        self.parser.add_argument("--embed_layer", nargs="+", type=int, help="scales used in the loss",
                                 default=[3, 4, 9])
        self.parser.add_argument("--frame_ids", nargs="+", type=int, help="frames to load", default=[0])
        self.parser.add_argument("--inchans", type=int, help="input image height", default=3)

        self.parser.add_argument("--img_scales", nargs="+", type=int, help="multiple scales for auxiliary loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--invert", help="set for invert depth estimation", action="store_true")
        self.parser.add_argument("--height", type=int, help="input image height", default=352)
        self.parser.add_argument("--model_name", type=str, help="the name of the folder to save the model in",
                                 default="Chitransformer")
        self.parser.add_argument("--png", help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--sa_depth", type=int, help="number of self attention layers", default=6)
        self.parser.add_argument("--scale", type=float, help="scale for invert depth", default=1.0)
        self.parser.add_argument("--shift", type=float, help="shift for invert depth", default=0.0)
        self.parser.add_argument("--width", type=int, help="input image width", default=1216)
        
        # training configurations:
        self.parser.add_argument("--batch_sampler", help="set batch sampler for dataloader", action='store_true')
        self.parser.add_argument("--batch_size", type=int, help="batch size", default=2)
        
        self.parser.add_argument("--check_period", type=int, help="number of epochs between each save", default=1)
        self.parser.add_argument("--clip_max_norm", type=float, help="clip_max_norm for grad updates", default=0.0)
        
        self.parser.add_argument("--data_path", type=str, help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--dataset", type=str, help="dataset to use", default="kitti",
                                 choices=["kitti", "cityscapes", "argoverse"])
        self.parser.add_argument("--device", type=str, help="set training device", default='cuda')
        
        self.parser.add_argument('--eval', help="set for evaluation", action="store_true")
        
        self.parser.add_argument("--freeze_embedder", help = "freeze the patch embedder during training",
                                 action="store_true")
        self.parser.add_argument("--freeze_dcr_ca", help = "freeze the DCR cross attention module",
                                 action="store_true")                      
        self.parser.add_argument("--frozen_weights", type=str, help="weight directory", default='')
        
        self.parser.add_argument("--learning_rate", type=float, help="set learning rate", default=1e-4)
        self.parser.add_argument("--learning_rate_pretrained", help="set learning rate for pretrained portion",
                                 type=float, default=1e-5)
        self.parser.add_argument("--lr_drop", type=int, help="set step size of the scheduler",default=20)
        self.parser.add_argument("--load_weights", type=str, help="directory of weights to load", default='')
        self.parser.add_argument("--log_freq", type=int, help="number of batches between each tensorboard log",
                                 default=10)
        
        self.parser.add_argument("--epochs", type=int, help="set number of epochs", default=30)
        self.parser.add_argument("--num_workers", type=int, help="number of dataloader workers", default=0)

        self.parser.add_argument('--only_dcr', help="set dcr only for fine-tuning", action="store_true") 

        self.parser.add_argument('--pre_pred', type=float, help="set for weight of guided loss in training", default=0.)

        self.parser.add_argument("--resume", type=str, help="resume the training process from a checkpoint", default='')
        
        self.parser.add_argument('--seed', type=int, help = "set seed for reproducibility", default=27)
        self.parser.add_argument("--source_scale", help="if set, losses are computed at the original scale",
                                 action="store_true")
        self.parser.add_argument("--split", type=str, help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "benchmark"], default="eigen_full")
        self.parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='start epoch')
        self.parser.add_argument('--supervision', type=float, help="set for weight of supervision loss for training", default=0.)
        self.parser.add_argument("--train_refinenet", help="train refinenet", action="store_true")
        self.parser.add_argument("--train_self_attention", help="train self-attention layer", action="store_true")
        
        self.parser.add_argument("--weight_decay", type=float, help="set weight_decay", default=0.)
        
        # ablation:
        self.parser.add_argument("--avg_reprojection", help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--edge_smoothness", help="set for edge aware depth smoothness",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking", help="if set, doesn't do auto-masking", action="store_true")
        self.parser.add_argument("--no_ssim", help="if set, disables ssim in the loss", action="store_true")
        self.parser.add_argument("--monocular", help = "set for monocular depth training", action="store_true")
        self.parser.add_argument('--rectilinear_epipolar_geometry', help="set for rectilinear images",
                                 action="store_true")
        self.parser.add_argument('--stereo', help="set stereo training", action="store_true")
        
        # training env:
        self.parser.add_argument("--dist_url", type=str, default='env://', help='url used to set up distributed training')
        self.parser.add_argument("--log_dir", type=str, help="log directory",
                                 default=os.path.join(os.path.expanduser("./"), "tmp"))                       
        self.parser.add_argument("--output_dir", type=str, help='folder to save checkpoints',
                                 default=os.path.join(file_dir, "output"))
        self.parser.add_argument("--world_size", type=int, default=1, help='number of distributed processes')       

    def parse(self):
        self.args = self.parser.parse_args()
        return self.args
