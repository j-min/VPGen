

import torch
from diffusers import StableDiffusionGLIGENPipeline
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageOps
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from viz_utils import plot_results, prepare_blank_image

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--layout_path', type=str,
                        )
    parser.add_argument('--image_dump_dir', type=str,
                        )
    parser.add_argument('--layout_image_dump_dir', type=str,
                        )
    parser.add_argument('--model', type=str, default='gligen', choices=['gligen', 'sd'])

    parser.add_argument('--n_bins', type=int, default=100)

    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--proc_id', type=int, default=0)

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # set seed
    from accelerate.utils import set_seed
    set_seed(args.seed)

    with open(args.layout_path, 'r') as f:
        layout_prediction_data = json.load(f)
    print('Loaded layout prediction data from', args.layout_path)

    layout_prediction_data = layout_prediction_data[:4]

    if args.n_proc > 1:
        # multi-processing
        # n_procs
        # proc_id
        print(f"Number of processes: {args.n_proc}")
        print(f"Process ID: {args.proc_id}")

        print(f"Number of total data points: {len(layout_prediction_data)}")
        layout_prediction_data = layout_prediction_data[args.proc_id::args.n_proc]
        print(f"Number of local data points: {len(layout_prediction_data)}")

    model_gen_dir = Path(args.image_dump_dir)
    model_gen_layout_dir = Path(args.layout_image_dump_dir)
    

    model_gen_dir.mkdir(parents=True, exist_ok=True)
    model_gen_layout_dir.mkdir(parents=True, exist_ok=True)


    print('Created dump directories')

    print('model_gen_dir:', model_gen_dir)
    print('model_gen_layout_dir:', model_gen_layout_dir)

    n_bins = args.n_bins

    if args.model == 'gligen':
        print('Loading GLIGEN model...')
        gligen_pipe = StableDiffusionGLIGENPipeline.from_pretrained(
            "gligen/diffusers-generation-text-box", revision="fp16", torch_dtype=torch.float16)
        pipe = gligen_pipe
    elif args.model == 'sd':
        print('Loading SD model...')
        original_SD_pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe = original_SD_pipe

    pipe.to('cuda')
                       

    def dummy(images, **kwargs):
        return images, False
    pipe.safety_checker = dummy
    print("Disabled safety checker")
    pipe.set_progress_bar_config(disable=True)

    desc = f"Proc ID: {args.proc_id} / {args.n_proc}:  - Generating images w/ {args.model}"

    for i, datum in tqdm(enumerate(layout_prediction_data),
                         desc=desc, total=len(layout_prediction_data),
                        ncols=50, dynamic_ncols=True):

        prompt = datum['caption']

        if 'id' in datum:
            image_fname = f"{datum['id']}.png"
        else:
            image_fname = f"{prompt}.png"

        model_gen_image_dump_path = model_gen_dir / image_fname
        model_gen_layout_dump_path = model_gen_layout_dir / image_fname
            

        if args.model == 'sd':
            
            generated_images = original_SD_pipe(prompt).images
            generated_image = generated_images[0]

        elif args.model in ['gligen']:
            phrases = []
            boxes = []

            if len(datum['objects']) == 0:
                print('No objects in datum')
                # continue
                # just use the prompt and full bounding box
                phrases = [prompt]
                boxes = [[0., 0., 1., 1.]]
            else:
                try:
                    for obj in datum['objects']:
                        box = obj['box']
                        x1, y1, x2, y2 = box

                        # de-quantize
                        x1 /= (n_bins - 1)
                        y1 /= (n_bins - 1)
                        x2 /= (n_bins - 1)
                        y2 /= (n_bins - 1)

                        phrases += [obj['text']]
                        boxes += [[x1, y1, x2, y2]]
                except:
                    print('Error in parsing boxes')
                    print('datum:', datum)
                    # print('obj:', obj)
                    # print('box:', box)
                    # print('x1, y1, x2, y2:', x1, y1, x2, y2)
                    # print('phrases:', phrases)
                    # print('boxes:', boxes)
                    # continue
                    phrases = [prompt]
                    boxes = [[0., 0., 1., 1.]]



            if args.model == 'gligen':
                generated_image = pipe(
                    prompt,
                    num_images_per_prompt=1,
                    gligen_phrases=phrases,
                    gligen_boxes=boxes,
                    gligen_scheduled_sampling_beta=0.3,
                    output_type="pil",
                    num_inference_steps=50,

                ).images[0]

        # Saving generated images
        generated_image.save(model_gen_image_dump_path)

        # Saving generated images + layouts
        if args.model in ['gligen']:

            # Add boounding box overlay to generated image    
            try:
                src_img = generated_image
                
                box_img = plot_results(
                    src_img,
                    boxes = boxes,
                    captions = phrases,
                    to_pil=True,
                )
                
                resize_H = 512
                resize_W = int(512 * box_img.size[0] / box_img.size[1])
                resize_size = (resize_W, resize_H)
                box_img = box_img.resize(resize_size)

                gen_image_box_overlay = ImageOps.expand(box_img, border=3, fill='black')
                gen_image_box_overlay.save(model_gen_layout_dump_path)

            except Exception as e:
                print('Error in plotting boxes')
                print('datum:', datum)
                print(e)