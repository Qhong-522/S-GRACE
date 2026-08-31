<div align="center">

# S-GRACE

## Rethinking Robust Adversarial Concept Erasure in Diffusion Models

### ECCV 2026

Qinghong Yin<sup>\*</sup>, Yu Tian<sup>\*</sup>, Heming Yang, Xiang Chen, Xianlin Zhang, Yue Ming, Xueming Li, and Yue Zhang<sup>†</sup>

<sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding author.

</div>

## Overview

S-GRACE (**S**emantic-**G**uided **R**obust **A**dversarial **C**oncept **E**rasure) is an efficient method for removing unsafe or unwanted concepts from text-to-image diffusion models. Existing adversarial concept erasure methods often face a trade-off between robustness and computational cost: adversarial optimization with only a few random samples may fail to capture the target concept space, while using more samples is expensive.

S-GRACE introduces two semantic-guided components:

- **Semantic-Guided Adversarial Optimization** uses a semantic prior to produce high-quality adversarial embeddings from a single sample.
- **Semantic-Guided Concept Erasure** trains the CLIP text encoder to map the target concept to a semantically similar surrogate while preserving unrelated concepts with anchor prompts.

Experiments cover nine concepts across NSFW content, artistic styles, and object categories. S-GRACE achieves strong robustness against adversarial attacks while preserving image generation utility, and erases one concept in approximately four minutes under the setup reported in the paper.

## Environment

The reference environment uses:

- Linux
- Python 3.12
- PyTorch 2.6.0 with CUDA 12.4
- `diffusers` 0.32.2
- `transformers` 4.49.0
- One NVIDIA GPU

Create the environment with Conda:

```bash
git clone https://github.com/Qhong-522/S-GRACE.git
cd S-GRACE

conda create -n s-grace python=3.12 -y
conda activate s-grace

pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install diffusers==0.32.2 transformers==4.49.0 accelerate==1.5.2 \
  peft==0.15.0 omegaconf==2.3.0 openai==1.69.0 \
  pillow==11.1.0 tqdm==4.67.1 numpy==2.2.4
```

The complete reference environment is recorded in `requirements.txt`. This file is a Conda package export rather than a standard pip requirements file, so do not use it with `pip install -r`.

## Model Preparation

The current release is configured for Stable Diffusion v1.4. Download the diffusion model and CLIP text encoder into the paths expected by `SD_PATH_TABLE` in `train.py`:

```bash
mkdir -p models

huggingface-cli download CompVis/stable-diffusion-v1-4 \
  --local-dir models/stable-diffusion-v1-4
huggingface-cli download openai/clip-vit-large-patch14 \
  --local-dir models/clip-vit-large-patch14
```

The resulting directory should look like this:

```text
models/
├── stable-diffusion-v1-4/
└── clip-vit-large-patch14/
```

If the models are stored elsewhere, update `SD_PATH_TABLE` near the top of `train.py`. You may need to log in with `huggingface-cli login` before downloading model weights.

## Quick Start

Run S-GRACE from the repository root with one of the provided configurations:

```bash
python train.py --config_file configs/church.yml --device cuda:0
```

The following experiment configurations are included:

| Category | Concepts | Configuration files |
| --- | --- | --- |
| Harmful content | Nudity, Violence, Illegal Activity | `nudity.yml`, `violence.yml`, `illegal_activity.yml` |
| Artistic style | Van Gogh, Picasso | `vangogh.yml`, `picasso.yml` |
| Object | Church, Parachute, Garbage Truck, Tench | `church.yml`, `parachute.yml`, `garbagetruck.yml`, `tench.yml` |

For example, to erase the Van Gogh style:

```bash
python train.py --config_file configs/vangogh.yml --device cuda:0
```

Each run creates a timestamped directory under the configured `output_path`, which is `outputs/train` by default:

```text
outputs/train/<timestamp>_<experiment>/
├── config.yml
├── logging.txt
├── all_adv_conditions.pt
├── edit_<step>_grid.jpg
└── text_encoder_<concept>_<run>/
```

The final subdirectory contains the erased CLIP text encoder in Hugging Face format.

## Using the Erased Text Encoder

Load the saved text encoder into a Stable Diffusion pipeline for inference:

```python
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel

base_model = "models/stable-diffusion-v1-4"
erased_encoder = "outputs/train/<run>/text_encoder_<concept>_<run>"

text_encoder = CLIPTextModel.from_pretrained(
    erased_encoder,
    torch_dtype=torch.float16,
)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    text_encoder=text_encoder,
    torch_dtype=torch.float16,
).to("cuda")

image = pipe("a photo of a cat").images[0]
image.save("result.png")
```

Replace `<run>` and `<concept>` with the names in your output directory.

## Custom Concepts and LLM Configuration

The `prompts/` directory already contains related and unrelated concepts for all provided experiment configurations, so an LLM API is **not required** to reproduce them.

For a new target concept:

1. Copy the closest configuration in `configs/` and update `exp_name`, `target_concept`, and `target_concept_type`.
2. Set `LLM_BASE_URL` and `LLM_API_KEY` near the top of `train.py` to an OpenAI-compatible endpoint.
3. Set `llm_model_name` in the new configuration.
4. Run `train.py`. If the corresponding prompt file does not exist, the related and unrelated concepts will be generated automatically and saved under `prompts/`.

The supported concept types are `style`, `character`, `object`, and `harmful`. Other training and testing options are documented directly in the YAML configurations.

## Repository Structure

```text
.
├── train.py          # S-GRACE training and concept-erasure pipeline
├── utils.py          # Diffusion, embedding, and image utilities
├── configs/          # Experiment configurations for the nine concepts
├── prompts/          # Related/unrelated concepts and test prompts
├── datas/            # Evaluation prompts used in the experiments
└── requirements.txt  # Full reference environment export
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yin2026sgrace,
  title     = {Rethinking Robust Adversarial Concept Erasure in Diffusion Models},
  author    = {Yin, Qinghong and Tian, Yu and Yang, Heming and Chen, Xiang and Zhang, Xianlin and Ming, Yue and Li, Xueming and Zhang, Yue},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
