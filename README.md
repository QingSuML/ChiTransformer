## ChiTransformer: Towards Reliable Stereo from Cues [CVPR 2022]

### [[cvpr](https://openaccess.thecvf.com/content/CVPR2022/html/Su_Chitransformer_Towards_Reliable_Stereo_From_Cues_CVPR_2022_paper.html) | [arXiv](https://arxiv.org/abs/2203.04554)]

[//]: # '<img width="1013" alt="image" src="https://user-images.githubusercontent.com/42019759/179230291-b3473a9c-763d-4776-9311-2f3de0d8d267.png">'

https://user-images.githubusercontent.com/42019759/179184580-aa4ef919-2f0c-4a56-a5d9-a42ef990d8d8.mp4

### Changelog
* [July 2022] Initial release of inference code and models



### Setup 

1) Download the model weights and place them in the `weights` folder:

- [dpt_hybrid-midas-501f0c75.pt](https://github.com/ISL-CV/Chi-Transformer/releases/download/1.0/chitransformer_kitti_301101.pt), [Mirror](https://drive.google.com/file/d/1_jLRcf96dnnzCz4F0pIQMvHy_TSVJ3p3/view?usp=sharing)


2) Set up dependencies: 

    ```shell
    pip install -r requirements.txt
    ```

   The code was tested with Python 3.9, PyTorch 1.8.0, OpenCV 4.5.1, and timm 0.4.5

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
   ```
    
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
