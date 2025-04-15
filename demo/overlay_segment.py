import gc
import multiprocessing as mp
import os
import time
from argparse import ArgumentParser
from functools import partial
from multiprocessing import cpu_count, Pool, Process
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from adhoc_image_dataset import AdhocImageDataset
from classes_and_palettes import GOLIATH_CLASSES, GOLIATH_PALETTE
from tqdm import tqdm
from worker_pool import WorkerPool
from overlay_utils import img_save_and_viz  # Import our overlay function

torchvision.disable_beta_transforms_warning()

timings = {}
BATCH_SIZE = 32


def warmup_model(model, batch_size):
    imgs = torch.randn(batch_size, 3, 1024, 768).to(dtype=model.dtype).cuda()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=model.dtype
    ):
        for i in range(3):
            model(imgs)
    torch.cuda.current_stream().wait_stream(s)
    imgs = imgs.detach().cpu().float().numpy()
    del imgs, s


def inference_model(model, imgs, dtype=torch.bfloat16, device="cuda"):
    with torch.no_grad():
        results = model(imgs.to(dtype).to(device))
        imgs = imgs.cpu()

    results = [r.cpu() for r in results]

    return results


def fake_pad_images_to_batchsize(imgs):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, BATCH_SIZE - imgs.shape[0]), value=0)


def load_model(checkpoint, use_torchscript=False):
    # if use_torchscript:
    return torch.jit.load(checkpoint)
    # else:
    #     return torch.export.load(checkpoint).module()


def main():
    parser = ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--input", help="Input image dir")
    parser.add_argument("--output_root", "--output-root", default=None, help="Path to output dir")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=4, help="Set batch size to do batch inference.")
    parser.add_argument("--shape", type=int, nargs="+", default=[1024, 768], help="input image size (height, width)")
    parser.add_argument("--fp16", action="store_true", default=False, help="Model inference dtype")
    parser.add_argument("--opacity", type=float, default=0.5, help="Opacity of painted segmentation map. In (0, 1] range.")
    parser.add_argument("--title", default="result", help="The image identifier.")
    parser.add_argument("--texture", default=None, help="Path to texture image for shirt overlay")
    args = parser.parse_args()

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("\033[93mCUDA is not available. Falling back to CPU.\033[0m")
        args.device = "cpu"

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    mp.log_to_stderr()
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.use_mixed_mm = True

    start = time.time()

    USE_TORCHSCRIPT = '_torchscript' in args.checkpoint

    # Build the model from a checkpoint file
    exp_model = load_model(args.checkpoint, USE_TORCHSCRIPT)

    # No precision conversion needed for TorchScript models; run at fp32
    if not USE_TORCHSCRIPT:
        dtype = torch.half if args.fp16 else torch.bfloat16
        exp_model.to(dtype)
        exp_model = torch.compile(exp_model, mode="max-autotune", fullgraph=True)
    else:
        dtype = torch.float32  # TorchScript models use float32

    exp_model = exp_model.to(args.device)  # Ensure model is on the correct device

    input = args.input
    image_names = []

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input  # Use the directory specified in input
        image_names = [
            image_name
            for image_name in sorted(os.listdir(input_dir))
            if image_name.endswith(".jpg")
            or image_name.endswith(".png")
            or image_name.endswith(".jpeg")
        ]
    elif os.path.isfile(input) and input.endswith(".txt"):
        with open(input, "r") as file:
            image_paths = [line.strip() for line in file if line.strip()]
        image_names = [os.path.basename(path) for path in image_paths]
        input_dir = os.path.dirname(image_paths[0]) if image_paths else ""
    else:
        raise ValueError("Invalid input, must be a directory or a text file")

    if len(image_names) == 0:
        raise ValueError("No images found in the input directory")

    # If left unspecified, create an output folder relative to input_dir.
    if args.output_root is None:
        args.output_root = os.path.join(input_dir, "output")

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    n_batches = (len(image_names) + args.batch_size - 1) // args.batch_size

    inference_dataset = AdhocImageDataset(
        [os.path.join(input_dir, img_name) for img_name in image_names],
        (input_shape[1], input_shape[2]),
        mean=[123.5, 116.5, 103.5],
        std=[58.5, 57.0, 57.5],
    )
    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(min(args.batch_size, cpu_count()), 1),
    )

    # Load the texture image if provided
    texture_img = None
    if args.texture:
        texture_img = cv2.imread(args.texture)
        if texture_img is None:
            raise ValueError("Texture image not found or unable to load: " + args.texture)

    total_results = []
    image_paths = []
    img_save_pool = WorkerPool(
        img_save_and_viz, processes=max(min(args.batch_size, cpu_count()), 1)
    )
    
    for batch_idx, (batch_image_name, batch_orig_imgs, batch_imgs) in tqdm(
        enumerate(inference_dataloader), total=len(inference_dataloader)
    ):
        valid_images_len = len(batch_imgs)
        batch_imgs = fake_pad_images_to_batchsize(batch_imgs)
        result = inference_model(exp_model, batch_imgs, dtype=dtype, device=args.device)

        args_list = [
            (
                i,
                r,
                os.path.join(args.output_root, os.path.basename(img_name)),
                GOLIATH_CLASSES,
                GOLIATH_PALETTE,
                args.title,
                args.opacity,
                texture_img,  # Pass texture image to the visualization function
            )
            for i, r, img_name in zip(
                batch_orig_imgs[:valid_images_len],
                result[:valid_images_len],
                batch_image_name,
            )
        ]
        for args_item in args_list:
            img_save_and_viz(*args_item)

    total_time = time.time() - start
    fps = 1 / ((time.time() - start) / len(image_names))
    print(f"\033[92mTotal inference time: {total_time:.2f} seconds. FPS: {fps:.2f}\033[0m")


if __name__ == "__main__":
    main()