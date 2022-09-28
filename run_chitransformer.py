import os
import glob
import torch
import cv2
import argparse
from functools import partial


from torchvision.transforms import Compose
from model.chitransformer import ChitransformerDepth
from model.dcr import DepthCueRectification_Sp
from utils.inference_utils import *



def run(input_path, output_path, model_path=None, optimize=True):
    net_w = 1216
    net_h = 352

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)
    
    model = ChitransformerDepth( 
                                device=device, 
                                dcr_module=partial(DepthCueRectification_Sp, 
                                layer_norm=False)
                                ).to(device)
    
    if model_path:
        checkpoint = torch.load(model_path, map_location='cpu')
        try:
            model.load_state_dict(checkpoint['model'])
        except:
            model.load_state_dict(checkpoint)

    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # get input
    assert os.path.isdir(os.path.join(input_path, "image_2")) and \
    os.path.isdir(os.path.join(input_path, "image_3")),\
    'Put left and right images in folder /image02 and /image03 respectively.'
    
    img_names_2 = glob.glob(os.path.join(input_path, "image_2", "*"))
    img_names_3 = glob.glob(os.path.join(input_path, "image_3", "*"))
    
    num_images = len(img_names_2)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    for ind, img_name_2 in enumerate(img_names_2):
        if os.path.isdir(img_name_2):
            img_fmt = img_name_2.split(".")[-1]
            if img_fmt in ['jpg', 'png']:
                continue
            
        name = img_name_2.split("/")[-1]
        
        img_name_3 = os.path.join(input_path, 'image_3', name)
        if os.path.isdir(img_name_3):
            continue

        print("  processing {} and {} ({}/{})".format(img_name_2, img_name_3, ind + 1, num_images))
        # input

        img_2 = read_image(img_name_2)
        img_3 = read_image(img_name_3)

        if args.kitti_crop is True:
            height, width, _ = img_2.shape
            top = height - 352
            left = (width - 1216) // 2
            img_2 = img_2[top : top + 352, left : left + 1216, :]
            img_3 = img_3[top : top + 352, left : left + 1216, :]

        img_2_input = transform({"image": img_2})["image"]
        img_3_input = transform({"image": img_3})["image"]

        # compute
        
        with torch.no_grad():
            img_2_input = torch.from_numpy(img_2_input).to(device).unsqueeze(0)
            img_3_input = torch.from_numpy(img_3_input).to(device).unsqueeze(0)
            
            if optimize == True and device == torch.device("cuda"):
                img_2_input = img_2_input.to(memory_format=torch.channels_last)
                img_3_input = img_3_input.to(memory_format=torch.channels_last)
                img_2_input = img_2_input.half()
                img_3_input = img_3_input.half()
                
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = model.forward(img_2_input, img_3_input)
            else:
                    prediction = model.forward(img_2_input, img_3_input)

            prediction = prediction[("depth", 0)]
            prediction = (
                            torch.nn.functional.interpolate(
                                prediction.unsqueeze(1),
                                size=img_2.shape[:2],
                                mode="bicubic",
                                align_corners=False,
                            )
                            .squeeze()
                            .cpu()
                            .numpy()
                        )

        filename = os.path.join(
                    output_path,
            os.path.splitext(os.path.basename(f'result_color_' + name.split('.')[0] + '.png'))[0]
                )
        write_depth_color(filename, 1/prediction+1e-8, absolute_depth=False)

    print("finished")

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="./inputs", help="folder with input images"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="./output",
        help="folder for output images",
    )

    parser.add_argument(
        "-m", "--model_weights", default="./weight", help="path to model weights"
    )

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")

    parser.add_argument("--optimize", dest="optimize", action="store_true")

    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.optimize,
    )
