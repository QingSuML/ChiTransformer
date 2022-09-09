<img width="1120" alt="predictions" src="https://user-images.githubusercontent.com/42019759/189385051-382d18db-1a98-4ee7-8295-0c73782e8260.png">
<img width="1120" alt="predictions2" src="https://user-images.githubusercontent.com/42019759/189411569-730f1980-3d63-4b4a-b446-1f6df209e4cc.png">

## ChiTransformer: Towards Reliable Stereo from Cues [CVPR 2022]
- #### The FIRST depth-cue-based binocular depth estimator.
- #### The binocular depth esitmator exempted from the long-standing ill-poisedness
- #### A fail-safe design that enables easy conversion between monocular and binocular estimation.

### Paper:
- [[cvpr2022 camera ready](https://openaccess.thecvf.com/content/CVPR2022/html/Su_Chitransformer_Towards_Reliable_Stereo_From_Cues_CVPR_2022_paper.html)]
- The latest version can be found at [arXiv](https://arxiv.org/abs/2203.04554)

https://user-images.githubusercontent.com/42019759/187740463-c49d9625-453b-4e9e-88d2-f06342765f1b.mp4

Monocular estimators are free from most of the ill-posedness faced by matching-based multi-view depth estimator. However, ,monocular depth estimators can not provide reliable depth prediction due to the lack of epipolar constraint. 

ChiTransformer is the first cue-based binocular depth estimator that leverages the strengths of both methods by introducing the Depth-Cue-Rectification (DCR) module to rectify the depth cues of two views in a cross-attention fashion under the underlying epipolar constraints that can be learned in DCR module. 


https://user-images.githubusercontent.com/42019759/187740535-17e2d477-b751-49fc-b356-51d6acfb4432.mp4

In the video above, Chitransformer is compared with the state-of-the-art monocular depth estimator, DPT-hybrid, to show the improvement in reliability, i.f.o object-depth consistency and relative-position-depth consistency. Visually significant improvements can be found where,   
- Tall cylindrical-like objects, e.g., Utiliy pole, light/sign poles, tall tree tunks,etc. The depth of this type of objects have near-uniform values from bottom to top. However, DPT always predicts the upper part of the object much closer, which, in the depth map, is much brigher. 
- The sky-line regions and the objects underneath. Though trained with dense gt and then fine-tuned on KITTI, DPT usually predicts the sky much closer to the foreground and predicts the objects underneath much further.
- Protruding, hanging-over objects, e.g., signboard, trailing plants, power-lines, etc. DPT tends to predict those objects closer.
- DPT shows some feature-copy behavior when there are structured features on the facades. Those features should not have that amount of depth difference with the surfaces they attach to.


### Changelog

* [Aug 2022] Implementation:
    - Please note that the parameters between the master and reference vit towers are shared.
    - Replace the reprojection W in polarized attention with the rounting mask
    - Hyperparameters for U in spectral decomposition changed due to different initialization method used.   
    - [**Highlight**] Practically, the DCR module of chitransformer can be a plug-and-go component to transform a attention-based monocular depth estimator (e.g. DPT) to a reliable cue-based binocular depth estimator within 3 epochs of extra training.


### Setup 

1) Download the model weights and place them in the `weights` folder:
- [chitransformer_kitti_15_03.pth](https://drive.google.com/file/d/1mpfFSZ5Zs8HZ6qk-FvqYAoSVY4gY9qn3/view?usp=sharing)  

    [Plug-and-Go model]:
- [chitransofrmer_plug_n_go.pth](https://drive.google.com/file/d/1KsKlJUgjdjamiFEFRTMxw_6REJmEZ-to/view?usp=sharing)   

2) Set up dependencies: 

    ```shell
    pip install -r requirements.txt
    ```

   - The code was tested with Python 3.9, PyTorch 1.11.0, OpenCV 4.5.5, and timm 0.5.4
   - **timm** of later version may not work due to function definition change.
   
   - **GPU memory** should be no smaller than 24GB.

### Usage 

- **Inference**

1) Place one or more input images in the folder `input`.

2) Run a stereo depth estimation model:

    ```shell
    python run_chitransformer.py -i ./inputs -o ./outputs -m [weight path] --kitti_crop --absolute_depth 
    ```


3) The results are written to the folder `output`.


- **Train**   

- To fine tune ChiTransformer on a dataset with or without ground truth:   

 ```shell
   torchrun --nproc_per_node=8 main.py --load_weight MODEL_PATH --data_path DATA_PATH --png --stereo --edge_smoothness --split SPLIT_TYPE --img_scales 0 --dcr_mode sp --rectilinear_epipolar_geometry [--freeze_embedder] [--only_dcr] [--train_refinenet] --epochs EPOCHS --lr_drop LR_DROP_POINT --learning_rate 0.00001 [--invert] [--crop]
   ```
Optional args should be set accordingly to achieve better performance. The current training pipeline is for (352, 1216) input. For input image of other sizes, you need to reconfigure accordingly. For more training options, please refer to $configs.py$.   

Training tips:
- Weights of loss components can be adjusted in the file `builder.py` in `build` function:
```
if args.dataset == "kitti":
        args.max_depth = 80.0
        args.min_depth = 1e-3
        
        if args.edge_smoothness:
            args.smoothness_weight = 0.1
            
        if args.dcr_mode in ["sp", "spectrum"]:
            weight_dict = {
                "reprojection_loss": 1.5,
                "orthog_reg": 0.1, 
                "hoyer_reg": 1e-3,
                "fp_loss" : 5e-5, 
                           }
            losses = [
                "reprojection_loss", 
                "orthog_reg", 
                "hoyer_reg", 
                "fp_loss",  
                    ]
``` 

- To facilitate a stable training process, a fixed intrinsic matrix of the reprojection-based
self-supervised training pipeline is desired. Make sure that the principle point is exactly centered due to the grid_sampler method used for image sampling.
- Direct fine-tuning on data with sparse ground truth will results in degraded details and depth-object and semantic consistency. To preserve the level of "descriptiveness" of the original model, it is recommended to run an inference and save the pre-prediction over the entire dataset prior the training. During training, the pre-prediction gradient can be leveraged through edge-aware smoothness loss to preserve the depth-object consistency.
    - Store the pre-prediction in the folder `/pred` under `data_path` folder.
    - You can specify `--pre_pred WEIGHT` to enable pre-pred guided training

### More qualitative comparisons with DPT

https://user-images.githubusercontent.com/42019759/187740352-7623a76b-3882-44db-8868-a6838c09d76e.mp4


<p align="left">
    <img src="https://user-images.githubusercontent.com/42019759/186490687-28468fac-4fbd-4a66-a421-cb3bcc17b5cf.png" width="740">
</p>

<p align="left">
    <img src="https://user-images.githubusercontent.com/42019759/186490733-446cd8e5-7f92-44ae-8009-59c26291ac8a.png" width="740">
</p>
<p align="left">
    Checkout the prediction on the whole sequence of KITTI_09_26_93 at: [https://youtu.be/WULXAFbuRqw]
</p>

<p align="left">
    <img src="https://user-images.githubusercontent.com/42019759/186490776-09e8e8c8-e130-4088-9280-aee5236fc763.png" width="740">
</p>
<p align="left">
    Checkout the prediction on the whole sequence of KITTI_10_03_34 at:[https://youtu.be/evif-Z8odYQ]
</p>


[**More videos**]
- Checkout the prediction on the whole sequence of KITTI_10_03_34 at: [https://youtu.be/evif-Z8odYQ]
- Checkout the prediction on the whole sequence of KITTI_09_26_104 at: [https://youtu.be/GbS2Sr6fMR0]
- Checkout the prediction on the whole sequence of KITTI_09_29_71 at: [https://youtu.be/2yUIVlZQmW4]
- Checkout the prediction on the whole sequence of KITTI_09_30_27 at: [https://youtu.be/hqtmCQF-w9g]


### Citation

Please cite this paper if you find the paper or code is useful.
```bibtex
@inproceedings{su2022chitransformer,
  title={ChiTransformer: Towards Reliable Stereo from Cues},
  author={Su, Qing and Ji, Shihao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1939--1949},
  year={2022}
}
```

### Acknowledgements

The work builds on and uses code from [DPT](https://github.com/isl-org/DPT.git), [Monodepth2](https://github.com/nianticlabs/monodepth2.git), [timm](https://github.com/rwightman/pytorch-image-models) and [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding). We'd like to thank the authors for making these libraries available.

### License 

MIT License 

