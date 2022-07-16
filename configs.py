
import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class TrainConfigs:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="ChiTransformer Training Configurations")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))
        self.parser.add_argument("--load_weights_folder",
                                type=str,
                                help="weight directory",
                                default='')

        # # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="Chitransformer")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "benchmark"],
                                 default="eigen_full")
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "cityscapes"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=352)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=1216)
        self.parser.add_argument("--device",
                                type=str,
                                help= "set training device",
                                default='cuda')
        self.parser.add_argument('--seed',
                                type=int,
                                help = "set seed for reproducibility",
                                default=42)
        self.parser.add_argument("--mono_use_stereo",
                                help="use stereo in monocular training",
                                action='store_true')
        self.parser.add_argument("--monocular",
                                help = "",
                                action="store_true")
        self.parser.add_argument('--stereo',
                                help="set stereo training",
                                action="store_true")
        self.parser.add_argument("--freeze_patch_embed",
                                help = "freeze the patch embedder during training",
                                action="store_true")
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--crop",
                                help="crop the input image",
                                action='store_true')
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--batch_sampler",
                                help="",
                                action='store_true')
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=1e-3)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0])
        """
        #     For stereo-supervised training:
        #     ``shell
        #     python train.py --model_name stereo_model \
        #     --frame_ids 0 --use_stereo --split eigen_full
        #     ```
            
        #     ArguementParser with narg='+' gathers all command-line args into a LIST.
        # """

        # # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=3)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--learning_rate_pretrained",
                                type=float,
                                default=1e-5)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=30)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=20)

        # # ABLATION options
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        # self.parser.add_argument("--weights_init",
        #                          type=str,
        #                          help="pretrained or scratch",
        #                          default="pretrained",
        #                          choices=["pretrained", "scratch"])


        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=0)

        # # LOADING options
        self.parser.add_argument("--weight_path",
                                 type=str,
                                 help="name of model to load")


        # # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=10)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")


        self.parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
        self.parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

       

    def parse(self):
        self.args = self.parser.parse_args()
        return self.args
