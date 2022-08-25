## ChiTransformer: Towards Reliable Stereo from Cues [CVPR 2022]
- ### The VERY FIRST depth-cue-based binocular depth estimator
- ### The binocular depth esitmator exempted from ill-poisedness
- ### A fail-safe design that enables easy conversion between monocular and binocular estimation.

### Paper:
- [[cvpr2022 camera ready](https://openaccess.thecvf.com/content/CVPR2022/html/Su_Chitransformer_Towards_Reliable_Stereo_From_Cues_CVPR_2022_paper.html)]
- The latest version can be found at [arXiv](https://arxiv.org/abs/2203.04554)

[//]: # '<img width="1013" alt="image" src="https://user-images.githubusercontent.com/42019759/179230291-b3473a9c-763d-4776-9311-2f3de0d8d267.png">'



https://user-images.githubusercontent.com/42019759/186087073-dcbd096c-e850-4c64-800b-88dc2d59ec7f.mp4   

Monocular estimators are free from most of the ill-posedness faced with matching-based multi-view depth estimator, however, ,monocular depth estimators can not provide reliable depth prediction due to the lack of epipolar constraint. ChiTransformer is the first cue-based binocular depth estimator that leverages the strengths of both methods by introducing the Depth-Cue-Rectification (DCR) module to rectify the depth cues of two views in a cross-attention fashion under the underlying epipolar constraints that can be learned in DCR module. 
In the video above, Chitransformer is compared with the state-of-the-art monocular depth estimator, DPT-hybrid to show the improvement in reliability, i.f.o content-depth consistency and relative-position-depth consistency. Visually significant improvements can be found where,   
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
- [chitransofrmer_model_ssgm0_5_07272022.pth](https://drive.google.com/file/d/1LIHNdyO8Jbhe0RWbpUzdbSo4zU82Tg41/view?usp=sharing)   


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
 

### More qualitative comparisons with DPT

![26_93_364_compare_1](https://user-images.githubusercontent.com/42019759/186490687-28468fac-4fbd-4a66-a421-cb3bcc17b5cf.png)
![26_93_364_compare_2](https://user-images.githubusercontent.com/42019759/186490733-446cd8e5-7f92-44ae-8009-59c26291ac8a.png)
Checkout the prediction on the whole sequence of KITTI_09_26_93 at: [https://youtu.be/Kr2pACAYWEc]

![1003_34_48_compare](https://user-images.githubusercontent.com/42019759/186490776-09e8e8c8-e130-4088-9280-aee5236fc763.png)
Checkout the prediction on the whole sequence of KITTI_10_03_34 at:[https://youtu.be/hEU8zkyIVPc]

[**More videos**]
- Checkout the prediction on the whole sequence of KITTI_10_03_34 at: [https://youtu.be/hEU8zkyIVPc]
- Checkout the prediction on the whole sequence of KITTI_09_26_104 at: [https://youtu.be/Ffn0nsLKrGk]
- Checkout the prediction on the whole sequence of KITTI_09_29_71 at: [https://youtu.be/Pehfnsm62FM]
- Checkout the prediction on the whole sequence of KITTI_09_30_27 at: [https://youtu.be/BAG7exlkaGI]


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

