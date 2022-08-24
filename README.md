## ChiTransformer: Towards Reliable Stereo from Cues [CVPR 2022]

#### [[cvpr](https://openaccess.thecvf.com/content/CVPR2022/html/Su_Chitransformer_Towards_Reliable_Stereo_From_Cues_CVPR_2022_paper.html)]
#### [The latest version can be found at [arXiv](https://arxiv.org/abs/2203.04554)]

[//]: # '<img width="1013" alt="image" src="https://user-images.githubusercontent.com/42019759/179230291-b3473a9c-763d-4776-9311-2f3de0d8d267.png">'



https://user-images.githubusercontent.com/42019759/186087073-dcbd096c-e850-4c64-800b-88dc2d59ec7f.mp4



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
   
   - **GPU memory** should be no smaller than 32GB.

### Usage 

**Inference**

1) Place one or more input images in the folder `input`.

2) Run a stereo depth estimation model:

    ```shell
    python run_chitransformer.py -i ./inputs -o ./outputs -m [weight path] --kitti_crop --absolute_depth 
    ```


3) The results are written to the folder `output`.


**Train**

The training pipeline is for (352, 1216) input. For input image of other sizes, you need to reconfigure accordingly. For more training options, please refer to configs.py.

   ```shell
   torchrun --nproc_per_node=8 main.py --crop --data_path [data pth] --png --stereo --split [split type]
   --other configurations
   ```
   
![26_93_364_compare_1](https://user-images.githubusercontent.com/42019759/186490687-28468fac-4fbd-4a66-a421-cb3bcc17b5cf.png)
    
![26_93_364_compare_2](https://user-images.githubusercontent.com/42019759/186490733-446cd8e5-7f92-44ae-8009-59c26291ac8a.png)

![1003_34_48_compare](https://user-images.githubusercontent.com/42019759/186490776-09e8e8c8-e130-4088-9280-aee5236fc763.png)

https://user-images.githubusercontent.com/42019759/186300034-573ff368-0d6e-4b43-b7fb-32a8bab0097c.mp4

Checkout the prediction on the whole sequence at: [https://youtu.be/Kr2pACAYWEc]

https://user-images.githubusercontent.com/42019759/186300065-7b752029-f1da-40a8-aabd-f813dd6f93ce.mp4

Checkout the prediction on the whole sequence at: [https://youtu.be/Ffn0nsLKrGk]

https://user-images.githubusercontent.com/42019759/186300089-9e7106b3-53d6-4494-a195-b62907de175c.mp4

Checkout the prediction on the whole sequence at: [https://youtu.be/Pehfnsm62FM]

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

