# coding: utf-8
import os
import gradio as gr
import random
import torch
import cv2
import re
import uuid
from PIL import Image, ImageDraw, ImageOps, ImageFont
import math
import numpy as np
import argparse
import inspect
import tempfile
import pandas as pd

from pathlib import Path

import json

from llama import LlamaHuggingFace
import task_utils
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Vicuna model
    parser.add_argument('--base_model', type=str, required=True, help='folder path to the vicuna with tokenizer')
    parser.add_argument('--lora_model', type=str, required=True, help='folder path to the lora model')
    
    # Sampling parameters
    parser.add_argument('--llm_device', type=str, default='cpu', help='device to run the llm model')
    # parser.add_argument('--temperature', type=float, default=0.1, help='temperature for the llm model')
    parser.add_argument('--temperature', type=float, default=0, help='temperature for the llm model')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='max number of new tokens to generate')
    parser.add_argument('--top_p', type=float, default=0.75, help='top_p for the llm model')
    parser.add_argument('--top_k', type=int, default=40, help='top_k for the llm model')
    parser.add_argument('--num_beams', type=int, default=1, help='num_beams for the llm model')
    parser.add_argument('--seed', type=int, default=42)

    # Multi-processing
    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--proc_id', type=int, default=0)

    # Data
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--skill', type=str, default='')
    parser.add_argument('--data_path', required=True, type=str, help='where to load prompts')

    # Where to save generated layouts
    parser.add_argument('--layout_dump_path', type=str, help='where to save generated layouts')

    args = parser.parse_args()
    print(args)

    if args.data == 'custom':
        with open(args.data_path, 'r') as f:
            example_skill2prompts = json.load(f)

        skills = list(example_skill2prompts.keys())
        prompts = []
        for skill in skills:
            print(f"Skill: {skill}") 
            print(f"Number of prompts: {len(example_skill2prompts[skill])}")
            prompts.extend(example_skill2prompts[skill])
        
        ids = [i for i in range(len(prompts))]

    elif args.data == 'vpeval':
        prompts = []
        ids = []
        with open(args.data_path, 'r') as f:
            data = json.load(f)['data']
            
        for datum in data:
            prompts.append(datum['text'])
            ids.append(datum['id'])

    elif args.data == 'tifa_human':
        # https://github.com/Yushi-Hu/tifa/blob/main/human_annotations/human_annotations.json

        with open(args.data_path, 'r') as f:
            data = json.load(f)

        ids = []
        prompts = []
        for k, v in data.items():
            if v['text_id'] in ids:
                pass
            else:
                ids.append(v['text_id'])
                prompts.append(v['text'])
            

    print(f"Number of Total prompts: {len(prompts)}")

    # Multi-processing
    # - Split prompts into 'n_proc' chunks and only use 'proc_id' chunk
    if args.n_proc > 1:
        print(f"Number of processes: {args.n_proc}")
        print(f"Process ID: {args.proc_id}")
        prompts = prompts[args.proc_id::args.n_proc]
        ids = ids[args.proc_id::args.n_proc]
        print(f"Number of local prompts: {len(prompts)}")

    load_dict = {}
    llm_kwargs = {'base_model': args.base_model,
                  'lora_model': args.lora_model,
                  'device': args.llm_device,
                  'temperature': args.temperature,
                  'max_new_tokens': args.max_new_tokens,
                  'top_p': args.top_p,
                  'top_k': args.top_k,
                  'num_beams': args.num_beams}
    
    print("Loading model...")
    from timeit import default_timer as timer
    start = timer()

    llm = LlamaHuggingFace(**llm_kwargs)
    model = llm

    end = timer()
    print(f"Time to load model: {end - start}")
    print("Model loaded successfully!")


    out_layouts = []

    seed_everything(args.seed)
    print(f"Seed: {args.seed}")

    desc = f"N process: {args.n_proc}, Process ID: {args.proc_id}, Generating layouts for {args.data}"

    for id, prompt in tqdm(zip(ids, prompts), total=len(prompts),
                           desc=desc):

        # Task 0
        source_text = "Instruction: Given an image caption, determine the objects and its counts to draw an image."
        source_text += "\n"
        source_text += "Caption: " + prompt

        print('Task 0')
        print(source_text)
        print()

        gen_text = model(source_text, None)[0]['generated_text']

        try:
            pred_objects = task_utils.decode_objects_from_text(gen_text)
        except:
            pred_objects = []

        print(gen_text)

        # Task 1
        source_text = "Instruction: Given an image caption and objects, determine the coordinates of the objects."
        source_text += "\n"
        source_text += "Caption: " + prompt
        source_text += "\n"
        source_text += "Objects: "
        obj_str = []
        for obj in pred_objects:
            obj_str += [f"{obj['text']} ({obj['count']})"]
        source_text += ' '.join(obj_str)

        print('Task 1')
        print(source_text)
        print()
        gen_text = model(source_text, None)[0]['generated_text']

        print(gen_text)

        try:
            pred_coordinates = task_utils.decode_coordinates_from_text(gen_text)
            # all box has 4 coordinates
            for objs in pred_coordinates:
                boxes = objs['boxes']
                for k, box in enumerate(boxes):
                    if len(box) > 4:
                        boxes[k] = box[:4]

            for objs in pred_coordinates:
                boxes = objs['boxes']
                for box in boxes:
                    assert len(box) == 4
        except Exception as e:
            print(pred_coordinates)
            print(e)
            pred_coordinates = []

        flatten_pred_coordinates = []
        for obj in pred_coordinates:
            for box in obj['boxes']:
                flatten_pred_coordinates.append({'text': obj['text'], 'box': box})
        print()

        out_layouts.append({
            'caption': prompt,
            'objects': flatten_pred_coordinates,
            'id': id
        })

    layout_dump_path = Path(args.layout_dump_path)
    layout_dump_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"Saving to {layout_dump_path}")

    with open(layout_dump_path, 'w') as f:
        json.dump(out_layouts, f, indent=4)
        print(f"Saved to {layout_dump_path}")