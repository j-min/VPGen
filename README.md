
# VPGen: Step-by-Step Text-to-Image Generation with Interpretable Visual Programming

The code for **VPGen**, a new framework for text-to-image generation, as described in the paper:

**[Visual Programming for Text-to-Image Generation and Evaluation](https://vp-t2i.github.io/)**

[Jaemin Cho](https://j-min.io),
[Abhay Zala](https://aszala.com/),
[Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

[[Project Page](https://vp-t2i.github.io)]
[[Paper](https://arxiv.org/abs/2305.15328)]
[[Code for VPEval](https://github.com/aszala/VPEval)]
[[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) Colab Demo](https://colab.research.google.com/github/j-min/VPGen/blob/main/VPGen_Demo.ipynb)]

<img width="800" src="assets/teaser_video.gif"/>

<br>

# VPGen: Step-by-Step T2I Generation

<img src="./assets/vpgen.png" width=1000px>

VPGen is a novel visual programming framework for interpretable step-by-step text-to-image (T2I) generation. As illustrated in the figure, we decompose the text-to-image generation task into three steps: (1) object/count generation, (2) layout generation, and (3) image generation. VPGen employs an LM to handle the first two steps: (1) object/count generation and (2) layout generation. Then VPGen uses a layout-to-image module to generate images from the predicted layouts. For the layout generation LM, we finetune Vicuna 13B on text-layout pair annotations on three public datasets: Flickr30K entities, MS COCO, and PaintSkills. For layout-to-image generation, we use GLIGEN.

<br>

# Code Structure

```bash
# Training & Inference Vicuna
utils/
task_utils.py
llama.py
lora_finetune.py
text2layout_inference.py

# Image inference with GLIGEN
inference_images.py
viz_utils.py
```

# Setup

## Setup Environment

```bash
conda create -n vpgen python=3.9
conda activate vpgen

pip install torch torchvision
pip install -r requirements.txt
```

## Setup Vicuna 13B

### Download pre-processed Vicuna 13B + LoRA Checkpoints from [HF Hub](https://huggingface.co/j-min/VPGen)

Currently we provide Vicuna13B + LoRA checkpoint finetuned on Flickr30K + COCO + PaintSkills. More checkpoints will be updated in the future.

```python
print("Installing HF hub")
# !pip install -q --upgrade huggingface_hub

print("Downloading Vicuna13B weights")

from huggingface_hub import snapshot_download
snapshot_download(repo_id="j-min/vicuna-13b-v0-merged",
                  repo_type="model",
                  local_dir="vicuna_13b_checkpoint",
                  force_download=True,
)

print("Downloading LoRA weights")

from huggingface_hub import hf_hub_download

for filename in ['adapter_config.json', 'adapter_model.bin']:
  hf_hub_download(repo_id="j-min/VPGen",
                  filename=filename,
                  subfolder="vicuna13B_GPU4_flickr30k_coco_paintskills_epoch2_mbatch32_lora16_cutoff256",
                  local_dir="CK/",
  )
```


### (Optional; Guideline to obtain Merged Vicuna 13B weights)

#### 1) Download LLama 13B checkpoint 

Weights for the LLaMA models can be obtained from by filling out [this form](http://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form).

#### 2) Convert the weights into Huggingface Transformers compatible version, following https://huggingface.co/docs/transformers/main/model_doc/llama.

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install -e .

python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights \
	--model_size 13B \
	--output_dir /output/path
```

#### 3) Download Vicuan 13B v0 delta weigths and merge with LLama weights to obtain Vicuna weights.

This conversion command needs around 60 GB of CPU RAM. See the "Low CPU Memory Conversion" section below if you do not have enough memory. Replace /path/to/* with the real paths.

Check https://github.com/lm-sys/FastChat#model-weights for more details.

```bash
# for v0 weights
pip install fschat==0.1.10

python -m fastchat.model.apply_delta \
    --base-model-path /path/to/llama-13b \
    --target-model-path vicuna_13b_checkpoint \
    --delta-path lmsys/vicuna-13b-delta-v0
```


## Setup GLIGEN

Check https://github.com/gligen/diffusers/tree/gligen/examples/gligen for more details.

```bash
git clone https://github.com/gligen/diffusers gligen_diffusers
cd gligen_diffusers
pip install -e .
```


# Finetuning Vicuna with LoRA

## Flickr30k+COCO+Paintskills training


```bash
n_gpus=4
model='vicuna13B'
base_model_path='vicuna_13b_checkpoint'

micro_batch_size=24
batch_size=96
lora_r=16
epochs=2
cutoff_len=512

# https://huggingface.co/j-min/VPGen/blob/main/flickr30k_coco_paintskills_text2box_train.json
data='flickr30k_coco_paintskills'

run_name=$model"_GPU$n_gpus"_epoch"$epochs"_mbatch"$micro_batch_size"_lora"$lora_r"_cutoff"$cutoff_len"
data_path='TRAIN_FILE'

torchrun --nproc_per_node=4 \
    lora_finetune.py \
	--base_model $base_model_path \
	--data_path $data_path \
	--output_dir './output/'$run_name \
	--prompt_template_name text2box \
	--num_epochs $epochs \
	--batch_size $batch_size \
	--cutoff_len $cutoff_len \
	--group_by_length \
	--lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
	--lora_r $lora_r \
	--micro_batch_size=$micro_batch_size
```


# Layout Inference with Vicuna

It takes 10-15 minutes to load Vicuna weights.
In our experiments, Vicuna 13B inference takes around 35GB CPU + 30GB GPU memory.

```bash
gpu_id=0

base_model_path='vicuna_13b_checkpoint'

# LoRA checkpoint path
lora_model_path='lora_checkpoint/vicuna13B_GPU4_flickr30k_coco_paintskills_epoch2_mbatch32_lora16_cutoff256'

# where to load prompts
prompts_path='DATA_PATH'

# Where to save the generated layouts
layout_dump_path='LAYOUT_DUMP_PATH'

echo $gpu_id

echo $base_model_path
echo $lora_model_path
echo $prompts_path
echo $layout_dump_path

python text2layout_inference.py \
	--llm_device "cuda:$gpu_id" \
	--base_model $base_model_path \
	--lora_model $lora_model_path \
	--data_path $prompts_path \
	--layout_dump_path $layout_dump_path
```

# Image Generation with GLIGEN

GLIGEN inference requires around 6GB of GPU RAM.

```bash
gpu_id=0

model='gligen'

# layout generated by Vicuna
layout_path='LAYOUT_DUMP_PATH'

# Where to save the images
image_dump_dir='IMAGE_DUMP_PATH'

# Where to save the bounding box images
layout_image_dump_dir='LAYOUT_IMAGE_DUMP_PATH'

echo $gpu_id

echo $layout_path
echo $image_dump_dir
echo $layout_image_dump_dir

CUDA_VISIBLE_DEVICES=$gpu_id \
python inference_images.py \
    --model $model \
    --layout_path $layout_path \
    --image_dump_dir $image_dump_dir \
    --layout_image_dump_dir $layout_image_dump_dir \
```


# Citation

If you find our project useful in your research, please cite the following paper:

```bibtex
@article{Cho2023VPT2I,
  author    = {Jaemin Cho and Abhay Zala and Mohit Bansal},
  title     = {Visual Programming for Text-to-Image Generation and Evaluation},
  year      = {2023},
}
```