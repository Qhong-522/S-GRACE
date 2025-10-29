import os
import PIL
import torch
import random
import logging
import numpy as np

from PIL import Image
from tqdm import tqdm
from typing import  Optional
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

def get_initial_latents(latent_batch_size, device, scheduler_init_noise_sigma, generator):
    return (torch.randn((latent_batch_size, 4, 64, 64), generator=generator).to(device)) * scheduler_init_noise_sigma

def tokenize(tokenizer, prompts):
    return tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

def detokenize(tokenizer, tokens):
    prompts = []
    for token in tokens:
        prompt = ''.join([tokenizer.decoder.get(t.item(), '') for t in token])
        prompts.append(prompt)
    return prompts

def ids_to_embeddings(text_encoder, input_ids):
    return text_encoder.text_model.embeddings(input_ids=input_ids)

def prompts_to_embeddings(tokenizer, text_encoder, prompts, device):
    input_ids = tokenize(tokenizer, prompts).input_ids.to(device)
    embeddings = text_encoder.text_model.embeddings(input_ids=input_ids)
    return embeddings

def prompts_to_conditions(tokenizer, text_encoder, prompts: list):
    input_ids = tokenize(tokenizer, prompts).input_ids.to(text_encoder.device)
    conditions_list = []
    for input_id in input_ids:
        conditions = text_encoder(input_id.unsqueeze(0))[0]
        conditions_list.append(conditions)
    conditions = torch.cat(conditions_list, dim=0)
    return conditions

def embeddings_to_conditions(text_encoder, embeddings):

    conditions_list = []
    for embedding in embeddings:
        
        bsz, tgt_len = (1, 77)
        mask = torch.full((tgt_len, tgt_len), torch.finfo(embedding.dtype).min, device=embedding.device)
        mask_cond = torch.arange(mask.size(-1), device=embedding.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(embedding.dtype)
        causal_attention_mask = mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

        encoder_outputs = text_encoder.text_model.encoder(
            inputs_embeds=embedding.unsqueeze(0),
            attention_mask=None,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        conditions = text_encoder.text_model.final_layer_norm(encoder_outputs[0])
        conditions_list.append(conditions)

    conditions = torch.cat(conditions_list, dim=0)
    return conditions

def conditions_to_features(projecter, embedding_init_length, conditions):
    return projecter(conditions[torch.arange(conditions.shape[0]), embedding_init_length])

def prompts_to_features(tokenizer, projecter, conditions, prompts):
    input_ids = tokenize(tokenizer, prompts).input_ids.to(conditions.device)
    features = projecter(conditions[torch.arange(conditions.shape[0]), input_ids.argmax(dim=-1)])
    return features

def get_clip_score(features1: torch.Tensor, features2: torch.Tensor):
    features1 = features1 / features1.norm(p=2, dim=-1, keepdim=True)
    features2 = features2 / features2.norm(p=2, dim=-1, keepdim=True)
    return features1 @ features2.T

def decode(vae, latents):
    return vae.decode(1 / vae.config.scaling_factor * latents).sample

def encode(vae, tensors):
    return vae.encode(tensors).latent_dist.mode() * 0.18215

def to_image(tensors):
    images = torch.stack([(tensors[i] / 2 + 0.5).clamp(0, 1) for i in range(tensors.shape[0])])
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images]

def set_scheduler_timesteps(scheduler, diffusion_scheduler_steps, device):
    scheduler.set_timesteps(diffusion_scheduler_steps, device=device)

def predict_noise(unet, scheduler, iteration, latents, conditions):
    if latents.shape[0] != conditions.shape[0]:
        raise ValueError(f"latents.shape[0] {latents.shape[0]} != conditions.shape[0] {conditions.shape[0]}")
    return unet(scheduler.scale_model_input(latents, iteration), iteration, encoder_hidden_states=conditions).sample

def diffusion(
    prompts: Optional[list],
    embeddings: Optional[torch.Tensor],
    negative_prompts: Optional[list],
    guidance_scale: float,
    diffusion_scheduler_steps: int,
    diffusion_end_steps: int,
    num_latents_per_condition: int,
    init_latents: Optional[torch.Tensor],
    generator: Optional[torch.Generator],
    return_type: str,
    show_progress: bool,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    vae: AutoencoderKL,
):
    device = unet.device
    
    if prompts is None and embeddings is None:
        raise ValueError("Either prompts or embeddings must be provided")

    if prompts is not None and embeddings is not None:
        logging.warning("Both prompts and embeddings are provided, only prompts will be used")

    if diffusion_end_steps is None or diffusion_end_steps > diffusion_scheduler_steps:
        diffusion_end_steps = diffusion_scheduler_steps
        logging.warning(f"diffusion_end_steps is None or diffusion_end_steps > diffusion_scheduler_steps, set diffusion_end_steps to diffusion_scheduler_steps {diffusion_scheduler_steps}")

    if prompts is not None:
        positive_conditions = prompts_to_conditions(tokenizer, text_encoder, prompts).repeat_interleave(num_latents_per_condition, dim=0)
        if negative_prompts is None:
            negative_prompts = [""] * len(prompts)
        if len(negative_prompts) != len(prompts):
            raise ValueError("If you want set negative_prompt for each prompt, please make sure the length of negative_prompts is equal to the length of prompts")
    else:
        positive_conditions = embeddings_to_conditions(text_encoder, embeddings).repeat_interleave(num_latents_per_condition, dim=0)
        negative_prompts = [""] * len(embeddings)
    negative_conditions = prompts_to_conditions(tokenizer, text_encoder, negative_prompts).repeat_interleave(num_latents_per_condition, dim=0)
    conditions = torch.cat([negative_conditions, positive_conditions])

    if init_latents is None:
        if generator is None:
            logging.warning("Both init_latents and generator are None, use default generator")
            generator = generator
        latents = get_initial_latents(num_latents_per_condition, unet.device, scheduler.init_noise_sigma, generator)
    else:
        latents = init_latents
    assert positive_conditions.shape[0] == negative_conditions.shape[0] == latents.shape[0], \
        f"Shape mismatch: positive_conditions.shape[0] = {positive_conditions.shape[0]}, " \
        f"negative_conditions.shape[0] = {negative_conditions.shape[0]}, " \
        f"init_latents.shape[0] = {latents.shape[0]}"

    set_scheduler_timesteps(scheduler, diffusion_scheduler_steps, unet.device)
    for diffusion_iteration in tqdm(scheduler.timesteps[0: diffusion_end_steps], disable=not show_progress):
        noise_pred = predict_noise(unet, scheduler, diffusion_iteration, torch.cat([latents] * 2), conditions)
        noise_pred_negative, noise_pred_positive = noise_pred.chunk(2)
        noise_pred = noise_pred_negative + guidance_scale * (noise_pred_positive - noise_pred_negative)
        latents = scheduler.step(noise_pred, diffusion_iteration, latents).prev_sample

    if return_type == 'latent':
        return latents
    
    latents_output = decode(vae, latents.to(device))
    
    if return_type == 'tensor':
        return latents_output
    
    images = to_image(latents_output)
    if return_type == 'image':
        return images
    else:
        raise ValueError(f"Invalid return type: {return_type}, either 'latent', 'tensor' or 'image'")

def save_images(images, save_folder, prompts, save_name_prefix='train', save_sep=True, save_grid=False):
    
    if not all(isinstance(img, Image.Image) for img in images):
        images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images]
        
    n_imgs = len(images) // len(prompts)
    
    if save_sep:
        for prompt_idx, prompt in enumerate(prompts):
            for img_idx, img in enumerate(images[prompt_idx * n_imgs: (prompt_idx + 1) * n_imgs]):
                img.save(os.path.join(save_folder, f"{save_name_prefix}_{prompt_idx}_{prompt[:10]}_{img_idx}.jpg"))
    
    if save_grid:
        n_rows = len(prompts)
        n_cols = n_imgs
        grid_size = (n_rows, n_cols)
        image_size = images[0].size
        grid_image = Image.new('RGB', (grid_size[1] * image_size[0], grid_size[0] * image_size[1]))
        for i, img in enumerate(images):
            grid_x = (i % grid_size[1]) * image_size[0]
            grid_y = (i // grid_size[1]) * image_size[1]
            grid_image.paste(img, (grid_x, grid_y))
        grid_image.save(os.path.join(save_folder, f"{save_name_prefix}_grid.jpg"))

@torch.no_grad()
def generate_images_with_embeddings(
    embeddings: torch.Tensor,
    diffusion_steps: int,
    save_prefix: str,
    num_images_per_prompt: int,
    test_batch_size: int,
    save_grid: bool,
    save_sep: bool,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    vae: AutoencoderKL,
    generator: torch.Generator,
    output_path: str,
    test_guidance_scale: float,
):
    logging.info(f"Generating images for {save_prefix}...")
    prompts = [f"emb_{i}" for i in range(len(embeddings))]
    
    test_batch_size = len(embeddings) if test_batch_size is None else test_batch_size

    init_latents = get_initial_latents(num_images_per_prompt, unet.device, scheduler.init_noise_sigma, generator)

    batched_embeddings = [embeddings[i: i + test_batch_size] for i in range(0, len(embeddings), test_batch_size)]

    all_images = []
    for batch in batched_embeddings:
        latents = init_latents.repeat(len(batch), 1, 1, 1)
        images = diffusion(
            prompts = None,
            embeddings = batch,
            negative_prompts = None,
            guidance_scale = test_guidance_scale,
            diffusion_scheduler_steps = diffusion_steps,
            diffusion_end_steps = diffusion_steps,
            num_latents_per_condition = num_images_per_prompt,
            init_latents = latents,
            generator = generator,
            return_type = 'image',
            show_progress = True,
            tokenizer = tokenizer,
            text_encoder = text_encoder,
            unet = unet,
            scheduler = scheduler,
            vae = vae,
        )
        for image in images:
            all_images.append(image)
    logging.info(f"{save_prefix} - generated {len(all_images)} images with embeddings")

    if save_grid or save_sep:
        save_images(
            images = all_images,
            save_folder = output_path,
            prompts = prompts,
            save_name_prefix = save_prefix,
            save_grid = save_grid,
            save_sep = save_sep
        )
        logging.info(f"{save_prefix} - test images saved at {output_path}")

@torch.no_grad()
def generate_images_with_prompts(
    prompts: list,
    diffusion_steps: int,
    save_prefix: str,
    num_images_per_prompt: int,
    test_batch_size: int,
    save_grid: bool,
    save_sep: bool,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    vae: AutoencoderKL,
    generator: torch.Generator,
    output_path: str,
    test_guidance_scale: float,
):
    logging.info(f"Generating images for {save_prefix} ...")
    
    test_batch_size = len(prompts) if test_batch_size is None else test_batch_size

    init_latents = get_initial_latents(num_images_per_prompt, unet.device, scheduler.init_noise_sigma, generator)

    batched_prompts = [prompts[i: i + test_batch_size] for i in range(0, len(prompts), test_batch_size)]

    all_images = []
    for batch in batched_prompts:
        latents = init_latents.repeat(len(batch), 1, 1, 1)
        images = diffusion(
            prompts = batch,
            embeddings = None,
            negative_prompts = None,
            guidance_scale = test_guidance_scale,
            diffusion_scheduler_steps = diffusion_steps,
            diffusion_end_steps = diffusion_steps,
            num_latents_per_condition = num_images_per_prompt,
            init_latents = latents,
            generator = generator,
            return_type = 'image',
            show_progress = True,
            tokenizer = tokenizer,
            text_encoder = text_encoder,
            unet = unet,
            scheduler = scheduler,
            vae = vae,
        )
        for image in images:
            all_images.append(image)
    logging.info(f"{save_prefix} - generated {len(all_images)}  with prompts")

    if save_grid or save_sep:
        save_images(
            images = all_images,
            save_folder = output_path,
            prompts = prompts,
            save_name_prefix = save_prefix,
            save_grid = save_grid,
            save_sep = save_sep
        )
        logging.info(f"{save_prefix} - test images saved at {output_path}")
        
    return all_images