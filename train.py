import os
import copy
import torch
import random
import logging
import argparse
import numpy as np

from tqdm import tqdm
from openai import OpenAI
from typing import Optional
from datetime import datetime
from omegaconf import OmegaConf

from peft import PeftModel, LoraConfig, get_peft_model
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler

from utils import *

import os
import time

LLM_BASE_URL = "" # provide your llm url
LLM_API_KEY = "" # provide your own key

# 4090
NOISE_PREDICT_BATCH_SIZE = 4
SD_PATH_TABLE = {
    'SD14': {
        'diffuser': 'models/stable-diffusion-v1-4',
        'encoder': 'models/clip-vit-large-patch14',
    },
    'SD15': {},
    'SD21': {},
    'SDXL': {},
    'FLUx': {},
}

def choose_parameters(parameter_choosing_strategy: str, text_encoder_train: CLIPTextModel):
                    
    logging.info(f"choosing parameters with strategy {parameter_choosing_strategy}...")
    parameters = []
    
    if parameter_choosing_strategy == "base":
        for name, param in text_encoder_train.text_model.named_parameters():
            if name.startswith('final_layer_norm') or name.startswith('encoder'):
                logging.info(f"{name}")
                parameters.append(param)
                
    elif parameter_choosing_strategy == "base_qkvo":
        for name, param in text_encoder_train.text_model.named_parameters():
            if name.endswith(('k_proj.weight', 'v_proj.weight', 'q_proj.weight', 'out_proj.weight')):
                logging.info(f"{name}")
                parameters.append(param)
    elif parameter_choosing_strategy == "base_qkvomlp":
        for name, param in text_encoder_train.text_model.named_parameters():
            if name.endswith(('k_proj.weight', 'v_proj.weight', 'q_proj.weight', 'out_proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight')):
                logging.info(f"{name}")
                parameters.append(param)
    elif parameter_choosing_strategy == "mlp":
        for name, param in text_encoder_train.text_model.named_parameters():
            if name.endswith(('mlp.fc1.weight','mlp.fc2.weight')):
                logging.info(f"{name}")
                parameters.append(param)
    elif parameter_choosing_strategy == "layer0":
        for name, param in text_encoder_train.text_model.named_parameters():
            if name.startswith('encoder.layers.0'):
                logging.info(f"{name}")
                parameters.append(param)

    elif parameter_choosing_strategy == "lora_qkvo":
        # text_encoder_train = get_peft_model(
        #     text_encoder_train, 
        #     LoraConfig(
        #         r = 8,
        #         lora_alpha = 16,
        #         init_lora_weights = "gaussian",
        #         target_modules = ["k_proj", "v_proj", "q_proj", "out_proj"]
        # ))
        # parameters.append(filter(lambda p: p.requires_grad, text_encoder_train.parameters())) 
        pass
    else:
        pass
    
    logging.info(f"Total number of training parameters: {sum(p.numel() for p in parameters)}")
    return parameters

def get_llm_answer(seed, model_name):
    logging.info(f"Asking llm {model_name}: {seed}")
    try:
        client = OpenAI(
            base_url = LLM_BASE_URL,
            api_key = LLM_API_KEY,
        )
        seed = seed
        answer = client.chat.completions.create(
            model = model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. You can help me by answering my questions. You can also ask me questions."},
                {"role": "user", "content": seed},
            ],
            temperature=0,
        ).choices[0].message.content
        logging.info(f"Got llm answer: {answer}")
        return answer
    except Exception as e:
        raise ValueError(f"Error get_llm_answer: {str(e)}")
    
def get_random_embedding(
        tokenizer,
        text_encoder,
        device,
        words_library,
        target_words_nums = 5,
        random_ids_nums = 5
    ):
    target_prompts = ",".join(random.sample(words_library, target_words_nums))
    target_input_ids = tokenize(tokenizer, target_prompts).input_ids.to(device)
    mid_target_input_ids = target_input_ids[0][1: target_input_ids.argmax(dim=-1)].unsqueeze(dim=0)
    
    if len(mid_target_input_ids[0]) + random_ids_nums >= 75:
        mid_target_input_ids = mid_target_input_ids[:, :-(len(mid_target_input_ids[0]) + random_ids_nums - 75)]
    
    embedding_length = len(mid_target_input_ids[0]) + random_ids_nums
    random_input_ids = torch.randint(0, 40905, (1, random_ids_nums), device = device)
    input_ids_pre = torch.tensor([49406], device = device).unsqueeze(dim=0)
    input_ids_post = torch.tensor([49407]*(tokenizer.model_max_length - 1 - embedding_length), device = device).unsqueeze(dim=0)
    input_ids = torch.cat([input_ids_pre, random_input_ids, mid_target_input_ids, input_ids_post], dim = 1)
    
    init_embedding = ids_to_embeddings(text_encoder, input_ids)
    return input_ids, init_embedding, embedding_length

def init_attack_embeddings_components(
        stragegy,
        tokenizer,
        text_encoder,
        device,
        words_library,
        embeddings_nums = 4,
        target_words_nums = 5,
        random_ids_nums = 5
    ):
    sot_embeddings, eot_embeddings = [], []
    att_embeddings, embeddings_length = [], []
    
    if stragegy == "copycat":
        for i in range(embeddings_nums):
            if i == 0:
                input_ids, init_embedding, embedding_length = get_random_embedding(
                    tokenizer,
                    text_encoder,
                    device,
                    [words_library[0]],
                    target_words_nums = 1,
                    random_ids_nums=0
                )
            else:
                input_ids, init_embedding, embedding_length = get_random_embedding(
                    tokenizer,
                    text_encoder,
                    device,
                    words_library,
                    target_words_nums = target_words_nums,
                    random_ids_nums = random_ids_nums
                )
            sot_embedding, att_embedding, eot_embedding = torch.split(
                init_embedding, 
                [1, embedding_length, tokenizer.model_max_length - 1 - embedding_length], 
                dim = 1
            )
            logging.info(f"Got init_embedding_{i} from: {detokenize(tokenizer, input_ids)}")
            sot_embedding = sot_embedding.detach()
            sot_embeddings.append(sot_embedding)
            eot_embedding = eot_embedding.detach()
            eot_embeddings.append(eot_embedding)
            att_embedding = att_embedding.detach()
            att_embeddings.append(att_embedding)
            embeddings_length.append(embedding_length)
    else:
        for i in range(embeddings_nums):
            embedding_length = 5
            att_random_input_ids = torch.randint(0, 40905, (1, embedding_length), device = device)
            target_input_ids = tokenize(tokenizer, [words_library[0]]).input_ids.to(device)
            mid_target_input_ids = target_input_ids[0][1: target_input_ids.argmax(dim=-1)].unsqueeze(dim=0)
            target_embedding_length = len(mid_target_input_ids[0])
            embedding_length = embedding_length + target_embedding_length
            input_ids_pre = torch.tensor([49406], device = device).unsqueeze(dim=0)
            input_ids_post = torch.tensor([49407]*(tokenizer.model_max_length - 1 - embedding_length), device = device).unsqueeze(dim=0)
            input_ids = torch.cat([input_ids_pre, att_random_input_ids, mid_target_input_ids, input_ids_post], dim = 1)
            init_embedding = ids_to_embeddings(text_encoder, input_ids)
            sot_embedding, att_embedding, eot_embedding = torch.split(
                init_embedding, 
                [1, embedding_length, tokenizer.model_max_length - 1 - embedding_length], 
                dim = 1
            )
            logging.info(f"Got init_embedding_{i} from: {detokenize(tokenizer, input_ids)}")
            sot_embedding = sot_embedding.detach()
            sot_embeddings.append(sot_embedding)
            eot_embedding = eot_embedding.detach()
            eot_embeddings.append(eot_embedding)
            att_embedding = att_embedding.detach()
            att_embeddings.append(att_embedding)
            embeddings_length.append(embedding_length)
    return sot_embeddings, eot_embeddings, att_embeddings, embeddings_length

def update_attack_embeddings_components(
        stragegy,
        sot_embeddings, 
        eot_embeddings,
        att_embeddings,
        embeddings_length,
        tokenizer,
        text_encoder,
        device,
        words_library,
        update_chances = 0.1,
        target_words_nums = 5,
        random_ids_nums = 5
    ):
    
    for i in range(len(sot_embeddings)):
        if random.random() < update_chances:
            if stragegy == "copycat":
                input_ids_, init_embedding_, embedding_length_ = get_random_embedding(
                    tokenizer,
                    text_encoder,
                    device,
                    words_library,
                    target_words_nums = target_words_nums,
                    random_ids_nums = random_ids_nums
                )
                sot_embedding_, att_embedding_, eot_embedding_ = torch.split(
                    init_embedding_, 
                    [1, embedding_length_, tokenizer.model_max_length - 1 - embedding_length_], 
                    dim = 1
                )
                logging.info(f"Got update_embedding_{i} from: {detokenize(tokenizer, input_ids_)}")
                sot_embedding_ = sot_embedding_.detach()
                sot_embeddings[i] = sot_embedding_
                eot_embedding_ = eot_embedding_.detach()
                eot_embeddings[i] = eot_embedding_
                att_embedding_ = att_embedding_.detach()
                att_embeddings[i] = att_embedding_
                embeddings_length[i] = embedding_length_
            else:
                att_random_input_ids = torch.randint(0, 40905, (1, embeddings_length[i]), device = device)
                target_input_ids = tokenize(tokenizer, [words_library[0]]).input_ids.to(device)
                mid_target_input_ids = target_input_ids[0][1: target_input_ids.argmax(dim=-1)].unsqueeze(dim=0)
                target_embedding_length = len(mid_target_input_ids[0])
                embedding_length = embeddings_length[i] + target_embedding_length
                embeddings_length[i] = embedding_length
                input_ids_pre = torch.tensor([49406], device = device).unsqueeze(dim=0)
                input_ids_post = torch.tensor([49407]*(tokenizer.model_max_length - 1 - embedding_length), device = device).unsqueeze(dim=0)
                input_ids = torch.cat([input_ids_pre, att_random_input_ids, mid_target_input_ids, input_ids_post], dim = 1)
                init_embedding = ids_to_embeddings(text_encoder, input_ids)
                sot_embedding, att_embedding, eot_embedding = torch.split(
                    init_embedding,
                    [1, embedding_length, tokenizer.model_max_length - 1 - embedding_length],
                    dim = 1
                )
                logging.info(f"Got update_embedding_{i} from: {detokenize(tokenizer, input_ids)}")
                sot_embedding = sot_embedding.detach()
                sot_embeddings[i] = sot_embedding
                eot_embedding = eot_embedding.detach()
                eot_embeddings[i] = eot_embedding
                att_embedding = att_embedding.detach()
                att_embeddings[i] = att_embedding
                embeddings_length[i] = embedding_length

    return sot_embeddings, eot_embeddings, att_embeddings, embeddings_length

def construct_embeddings_from_components(
        sot_embeddings,
        eot_embeddings,
        att_embeddings,
    ):
    embeddings = []
    for sot_embedding, eot_embedding, att_embedding in zip(sot_embeddings, eot_embeddings, att_embeddings):
        embedding = torch.cat([sot_embedding, att_embedding, eot_embedding], dim = 1)
        embeddings.append(embedding)
    return torch.cat(embeddings, dim = 0)   

def get_loss_l2(tensor1: torch.Tensor, tensor2: torch.Tensor, reduction: Optional[str] = "mean"):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Input tensors must have the same shape for l2 loss.")
    return torch.nn.MSELoss(reduction = reduction)(tensor1, tensor2)

def get_loss_clip(tensor1: torch.Tensor, tensor2: torch.Tensor, reduction: Optional[str] = "mean"):
    if reduction == 'mean':
        return torch.mean(get_clip_score(tensor1, tensor2))
    elif reduction == 'row':
        if tensor1.shape[0] != tensor2.shape[0]:
            raise ValueError("Input tensors must have the same shape for row clip loss")
        loss = 0
        for i in range(tensor1.shape[0]):
            loss += get_clip_score(tensor1[i].unsqueeze(0), tensor2[i].unsqueeze(0))
        loss = loss / tensor1.shape[0]
        return loss

def train_attack_embeddings(
        embeddings_type, # ["adv", "dis"]
        attack_type,
        steps,
        learning_rate,
        sot_embeddings,
        eot_embeddings,
        att_embeddings_0,
        embeddings_length,
        embeddings_nums,
        time_steps,
        init_noisies,
        noisy_images,
        text_encoder_train,
        unet,
        scheduler,
        device,
        global_steps,
        **kwargs,
    ):
    lr = learning_rate
    att_embeddings = []
    optimizers = []
    embeddings_max_value = []
    embeddings_min_value = []
    for att_embedding_0 in att_embeddings_0:
        max_value, min_value = att_embedding_0.max(), att_embedding_0.min()
        embeddings_max_value.append(max_value)
        embeddings_min_value.append(min_value)
        att_embedding = torch.nn.Parameter(att_embedding_0.to(device))
        optimizer = torch.optim.Adam([att_embedding], lr = learning_rate)
        att_embeddings.append(att_embedding)
        optimizers.append(optimizer)
    logging.info(f"Start {embeddings_type} embedding learning for global steps {global_steps}, embeddings nums {embeddings_nums}, learning_rate: {lr}.")
    
    for step in range(steps):
    
        for optimizer in optimizers:
            optimizer.zero_grad()
        
        # loss1
        input_embeddings = construct_embeddings_from_components(
            sot_embeddings = sot_embeddings,
            eot_embeddings = eot_embeddings,
            att_embeddings = att_embeddings,
        ).repeat_interleave(time_steps.shape[0] // embeddings_nums, dim = 0)
        total_loss = 0
        for batch_start in range(0, input_embeddings.shape[0], NOISE_PREDICT_BATCH_SIZE):
            batch_end = min(batch_start + NOISE_PREDICT_BATCH_SIZE,  input_embeddings.shape[0])

            input_embedding = input_embeddings[batch_start: batch_end]
            init_noisy = init_noisies[batch_start: batch_end]
            noisy_image = noisy_images[batch_start: batch_end]
            time_step = time_steps[batch_start: batch_end]

            input_conditions = embeddings_to_conditions(text_encoder_train, input_embedding)
            scores = predict_noise(unet, scheduler, time_step, noisy_image, input_conditions)
            loss = get_loss_l2(init_noisy.to(device), scores.to(device), reduction="mean")
            loss = loss * (batch_end - batch_start) / input_embeddings.shape[0]
            loss.backward()
            total_loss += loss.item()
        logging.info(f"global steps {global_steps}, {embeddings_type} embedding learning step {step}, loss: {total_loss}")

        # loss2
        if kwargs["use_clip_target_sim"]:
            input_adv_embeddings = construct_embeddings_from_components(
                sot_embeddings = sot_embeddings,
                eot_embeddings = eot_embeddings,
                att_embeddings = att_embeddings,
            )
            input_adv_conditions = embeddings_to_conditions(text_encoder_train, input_adv_embeddings)
            features_adv = conditions_to_features(kwargs["projecter"], embeddings_length, input_adv_conditions)
            loss_tar_sim_reg = get_loss_clip(features_adv, kwargs["features_target"])
            clip_dis = loss_tar_sim_reg.item()
            
            if embeddings_type == "adv":
                loss_tar_sim_reg = loss_tar_sim_reg * -1 + 1
            elif embeddings_type == "dis":
                loss_tar_sim_reg = loss_tar_sim_reg + 1
                
            loss_tar_sim_reg = loss_tar_sim_reg * kwargs["clip_target_sim_weight"]
            loss_tar_sim_reg.backward()
            logging.info(f"global steps {global_steps}, {embeddings_type} embedding learning step {step}, loss_tar_sim_reg: {loss_tar_sim_reg.item()}, (clip dis = {clip_dis})")
        
        for att_embedding in att_embeddings:
            att_embedding.grad.sign_()
        for optimizer in optimizers:
            optimizer.step()
        if attack_type == "pgd":
            for att_embedding_0, att_embedding, max_value, min_value in \
                zip(att_embeddings_0, att_embeddings, embeddings_max_value, embeddings_min_value):
                delta = torch.clamp(att_embedding - att_embedding_0, min = -lr * 4, max = lr * 4)
                att_embedding = torch.clamp(att_embedding_0 + delta, min=max_value, max=min_value).detach()
        elif attack_type == "fgsm":
            for att_embedding_0, att_embedding in zip(att_embeddings_0, att_embeddings):
                att_embedding_0 = att_embedding
                                
    embeddings_result = construct_embeddings_from_components(
        sot_embeddings = sot_embeddings,
        eot_embeddings = eot_embeddings,
        att_embeddings = att_embeddings,
    ).detach()
    
    return embeddings_result
    
def train(conf: OmegaConf, save_folder: str):
    
    target_concept = conf.target_concept
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(conf.seed_train)
    use_test = conf.test.switch_on
    
    # loading test prompts
    test_prompts_path = f"prompts/{conf.target_concept.replace(' ', '_')}_test.txt"
    if os.path.exists(test_prompts_path):
        try:
            with open(test_prompts_path, 'r') as file:
                test_prompts = [prompt.strip() for prompt in file.readlines()]
        except Exception as e:
            raise ValueError(f"An error occurred while loading test prompts: {e}")
    else:
        test_prompts = [conf.target_concept, "a photo of a cat", "a photo of a car"]
    logging.info(f"Loaded {len(test_prompts)} test prompts.")
    for i, prompt in enumerate(test_prompts):
        logging.info(f"Test prompt {i + 1}: {prompt}")
        
    # loading related concepts and unrelated concepts
    related_prompts, unrelated_prompts = [], []
    related_prompts.append(target_concept)
    # for _ in range(conf.llm_knowledge_related_concepts_nums // 20):
    #     related_prompts.append(target_concept)
    llm_model_name = conf.llm_model_name
    llm_knowledge_path = f"prompts/{target_concept.replace(' ', '_')}_{llm_model_name}_related_{conf.llm_knowledge_related_concepts_nums}_unrelated_{conf.llm_knowledge_unrelated_concepts_nums}.txt"
    if not os.path.exists(llm_knowledge_path):
        logging.info(f"llm knowledge prompts not found in {llm_knowledge_path}, use gpt to generate it.") 
        related_raw_prompts = get_llm_answer(
            seed = f"Please give me {conf.llm_knowledge_related_concepts_nums} keywords describing {target_concept} or you believe are very related to {target_concept}, separated by comma. Start your response directly.",
            model_name = llm_model_name
        )
        unrelated_raw_prompts = get_llm_answer(
            seed = f"Please give me {conf.llm_knowledge_unrelated_concepts_nums} keywords often appeared with {target_concept} but unrelated to {target_concept}, separated by comma. Start your response directly.",
            model_name = llm_model_name
        )
        for raw_prompt in related_raw_prompts.split(','):
            related_prompts.append(raw_prompt.strip())
        for raw_prompt in unrelated_raw_prompts.split(','):
            unrelated_prompts.append(raw_prompt.strip())
        logging.info(f"Saving llm knowledge prompts to {llm_knowledge_path}.")
        with open(llm_knowledge_path, "w") as file:
            for prompt in related_prompts:
                file.write(prompt + '\n') 
            file.write('\n')
            for prompt in unrelated_prompts:
                file.write(prompt + '\n') 
    else:
        logging.info(f"llm knowledge found in {llm_knowledge_path}, loading it directly from file.")
        with open(llm_knowledge_path, "r", encoding="utf-8") as file:
            cur_prompts = related_prompts
            for line in file.readlines():
                if line == "\n":
                    cur_prompts = unrelated_prompts
                else:
                    cur_prompts.append(line.strip())       
    logging.info(f"Loaded {len(related_prompts)} related prompts.")
    for i, prompt in enumerate(related_prompts):
        logging.info(f"Related prompt {i + 1}: {prompt}")
    logging.info(f"Loaded {len(unrelated_prompts)} unrelated prompts.")
    for i, prompt in enumerate(unrelated_prompts):
        logging.info(f"Unelated prompt {i + 1}: {prompt}")
    
    # loading model...
    logging.info("Loading model...")
    if conf.model_version not in SD_PATH_TABLE:
        raise ValueError(f"Model version {conf.model_version} not supported. Available versions are: {', '.join(SD_PATH_TABLE.keys())}.")
    diffusion_model_path = SD_PATH_TABLE[conf.model_version]['diffuser']
    text_encoder_model_path = SD_PATH_TABLE[conf.model_version]['encoder']
    
    vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae").eval().to(device)
    unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet").eval().to(device)
    scheduler = DDIMScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")
    
    tokenizer = CLIPTokenizer.from_pretrained(text_encoder_model_path)
    projecter = CLIPModel.from_pretrained(text_encoder_model_path).text_projection.eval().to(device)
    text_encoder_freeze = CLIPTextModel.from_pretrained(text_encoder_model_path).eval().to(device)
    
    text_encoder_train = CLIPTextModel.from_pretrained(diffusion_model_path, subfolder='text_encoder').train().to(device)          
    
    # testing original model on test prompts
    if use_test and conf.test.test_original:
        generate_images_with_prompts(
            prompts = test_prompts,
            diffusion_steps = conf.test.sample_steps,
            save_prefix = "original",
            num_images_per_prompt = conf.test.test_images_num_per_prompt,
            test_batch_size = NOISE_PREDICT_BATCH_SIZE // conf.test.test_images_num_per_prompt,
            save_grid = True,
            save_sep = False,
            tokenizer = tokenizer,
            text_encoder = text_encoder_freeze,
            unet = unet,
            scheduler = scheduler,
            vae = vae,
            generator = torch.Generator().manual_seed(conf.test.seed),
            output_path = save_folder,
            test_guidance_scale = conf.test.guidance_scale
        )

    # choosing parameters
    training_parameters = choose_parameters(conf.parameter_choosing_strategy, text_encoder_train)
    if conf.reg.param.switch_on:
        original_training_parameters_copy = [param.clone().detach() for param in training_parameters]
    
    with torch.no_grad():
        # kits for clip distance with target
        conditions_target = prompts_to_conditions(
            tokenizer = tokenizer,
            text_encoder = text_encoder_freeze,
            prompts = target_concept,
        ).detach()
        features_target = prompts_to_features(tokenizer, projecter, conditions_target, target_concept).detach()
    
    # preparing training kits
    if conf.target_concept_type == "style":
        target_concept = [f"a picture of {conf.target_concept}"]
        # target_concept = [f"{conf.target_concept}"]
    elif conf.target_concept_type == "object" or conf.target_concept_type == "character":
        target_concept = [f"a photo of {conf.target_concept}"]
    elif conf.target_concept_type == "harmful":
        target_concept = [f"{conf.target_concept}"]
    else:
        raise ValueError(f"target concept type {conf.target_concept_type} not supported")
    logging.info(f"Start erasing target concept: {target_concept}")
    
    # =========================================================================================================
    # =========================================================================================================
    # ===========================================Start Traning...==============================================
    # =========================================================================================================
    # =========================================================================================================
    
    erase_time_start = time.time()
    total_adv_time = 0.0  

    global_steps = 0
    edit_optimizer = torch.optim.Adam(training_parameters, lr = conf.edit.learning_rate)
    all_adv_conditions = []
    all_dis_conditions = []

    # repeat attack-unlearn-attack-unlearn... rounds
    for training_round in range(conf.training_rounds):

        adv_time_start = time.time()
        
        # ================================== learning adv embeddings ↓ ==================================
        adv_embeddings_nums = conf.adv.embeddings_nums
        
        if global_steps == 0:
            adv_sot_embeddings, adv_eot_embeddings, adv_att_embeddings_0, adv_embeddings_length = init_attack_embeddings_components(
                stragegy = conf.adv.init_stragegy,
                tokenizer = tokenizer,
                text_encoder = text_encoder_freeze,
                device = device,
                words_library = related_prompts,
                embeddings_nums = adv_embeddings_nums,
                target_words_nums = random.randint(conf.adv.init_words_range[0], conf.adv.init_words_range[1]),
                random_ids_nums = conf.adv.random_ids_nums
            )
        else:
            adv_sot_embeddings, adv_eot_embeddings, adv_att_embeddings_0, adv_embeddings_length = update_attack_embeddings_components(
                stragegy = conf.adv.init_stragegy,
                sot_embeddings = adv_sot_embeddings, 
                eot_embeddings = adv_eot_embeddings,
                att_embeddings = adv_att_embeddings_0,
                embeddings_length = adv_embeddings_length,
                tokenizer = tokenizer,
                text_encoder = text_encoder_freeze,
                device = device,
                words_library = related_prompts,
                update_chances = conf.adv.update_chance,
                target_words_nums = random.randint(conf.adv.init_words_range[0], conf.adv.init_words_range[1]),
                random_ids_nums = conf.adv.random_ids_nums
            ) 
        
        if conf.reg.dis.switch_on:
            dis_sot_embeddings, dis_eot_embeddings, dis_att_embeddings_0, dis_embeddings_length = \
                copy.deepcopy(adv_sot_embeddings), copy.deepcopy(adv_eot_embeddings), \
                copy.deepcopy(adv_att_embeddings_0), copy.deepcopy(adv_embeddings_length)
               
        if global_steps == 0 or not conf.few_shot:
            with torch.no_grad():
                init_noisies_list, noisy_images_list, time_steps_list = [], [], []
                init_noisies = get_initial_latents(conf.edit.dis_noisy_img_nums, device, scheduler.init_noise_sigma, generator)
                random_ddpm_time_steps = random.sample(range(conf.edit.ddpm_time_step_low, conf.edit.ddpm_time_step_high), conf.edit.dis_time_step_nums)
                set_scheduler_timesteps(scheduler, conf.edit.scheduler_steps, device)
                time_steps_tensor_list = []
                for time_step in random_ddpm_time_steps:
                    # time_step = int(time_step / 1000 * conf.edit.scheduler_steps) * (1000 / conf.edit.scheduler_steps)
                    time_steps_tensor_list.append(torch.tensor((time_step,), device = device))
                logging.info(f"Using time step: {time_steps_tensor_list}")
                for time_step in time_steps_tensor_list:
                    noisy_images = diffusion(
                        prompts = target_concept,
                        embeddings = None,
                        negative_prompts = None,
                        guidance_scale = conf.edit.guidance_scale,
                        diffusion_scheduler_steps = conf.edit.scheduler_steps,
                        diffusion_end_steps = int(time_step.item() / 1000 * conf.edit.scheduler_steps),
                        num_latents_per_condition = init_noisies.shape[0],
                        init_latents = init_noisies,
                        generator = generator,
                        return_type = 'latent',
                        show_progress = True,
                        tokenizer = tokenizer,
                        text_encoder = text_encoder_freeze,
                        unet = unet,
                        scheduler = scheduler,
                        vae = vae,
                    )
                    for init_noise, noisy_image in zip(init_noisies, noisy_images):
                        init_noisies_list.append(init_noise.unsqueeze(dim=0).detach())
                        noisy_images_list.append(noisy_image.unsqueeze(dim=0).detach())
                        time_steps_list.append(time_step.detach())
                init_noisies_0 = torch.cat(init_noisies_list, dim = 0)
                noisy_images_0 = torch.cat(noisy_images_list, dim = 0)
                time_steps_0 = torch.cat(time_steps_list, dim = 0)
                set_scheduler_timesteps(scheduler, 1000, device)
                adv_init_noisies = init_noisies_0.repeat(adv_embeddings_nums, 1, 1, 1)
                adv_noisy_images = noisy_images_0.repeat(adv_embeddings_nums, 1, 1, 1)
                adv_time_steps = time_steps_0.repeat(adv_embeddings_nums)
        
        adv_embeddings_result = train_attack_embeddings(
            embeddings_type = "adv",
            attack_type = conf.adv.attack_type,
            steps = conf.adv.steps,
            learning_rate = conf.adv.learning_rate,
            sot_embeddings = adv_sot_embeddings,
            eot_embeddings = adv_eot_embeddings,
            att_embeddings_0 = adv_att_embeddings_0,
            embeddings_length = adv_embeddings_length,
            embeddings_nums = adv_embeddings_nums,
            time_steps = adv_time_steps,
            init_noisies = adv_init_noisies,
            noisy_images = adv_noisy_images,
            text_encoder_train = text_encoder_train,
            unet = unet,
            scheduler = scheduler,
            device = device,
            global_steps = global_steps,
            use_clip_target_sim = conf.adv.reg.clip_target_sim.switch_on,
            projecter = projecter,
            features_target = features_target,
            clip_target_sim_weight = conf.adv.reg.clip_target_sim.weight
        )
        
        with torch.no_grad():
            conditions_adv_0 = embeddings_to_conditions(text_encoder_train, adv_embeddings_result)
            if args.save_features:
                all_adv_conditions.append(conditions_adv_0)  # 存储 adv 样本的 condition
            features_adv_0 = conditions_to_features(projecter, adv_embeddings_length, conditions_adv_0).detach()
            
         
        if use_test and conf.test.test_adv_embeddings_before_edit:
            generate_images_with_embeddings(
                embeddings = adv_embeddings_result,
                diffusion_steps = conf.test.sample_steps,
                save_prefix = f"embeddings_{global_steps}_adv_before",
                num_images_per_prompt = conf.test.test_images_num_per_emb,
                test_batch_size = NOISE_PREDICT_BATCH_SIZE // conf.test.test_images_num_per_emb,
                save_grid = True,
                save_sep = False,
                tokenizer = tokenizer,
                text_encoder = text_encoder_train,
                unet = unet,
                scheduler = scheduler,
                vae = vae,
                generator = torch.Generator().manual_seed(conf.test.seed),
                output_path = save_folder,
                test_guidance_scale = conf.test.guidance_scale
            )
        # ================================== learning adv embeddings ↑ ==================================
        if conf.reg.dis.switch_on:
            dis_embeddings_result = train_attack_embeddings(
                embeddings_type = "dis",
                attack_type = conf.adv.attack_type,
                steps = conf.adv.steps,
                learning_rate = conf.adv.learning_rate,
                sot_embeddings = dis_sot_embeddings,
                eot_embeddings = dis_eot_embeddings,
                att_embeddings_0 = dis_att_embeddings_0,
                embeddings_length = dis_embeddings_length,
                embeddings_nums = adv_embeddings_nums,
                time_steps = adv_time_steps,
                init_noisies = adv_init_noisies,
                noisy_images = adv_noisy_images,
                text_encoder_train = text_encoder_train,
                unet = unet,
                scheduler = scheduler,
                device = device,
                global_steps = global_steps,
                use_clip_target_sim = True,
                projecter = projecter,
                features_target = features_target,
                clip_target_sim_weight = conf.adv.reg.clip_target_sim.weight
            )
            with torch.no_grad():
                conditions_dis_0 = embeddings_to_conditions(text_encoder_train, dis_embeddings_result)
                if args.save_features:
                    all_dis_conditions.append(conditions_dis_0)  
            if use_test and conf.test.test_dis_embeddings_before_edit:
                generate_images_with_embeddings(
                    embeddings = dis_embeddings_result,
                    diffusion_steps = conf.test.sample_steps,
                    save_prefix = f"embeddings_{global_steps}_dis_before",
                    num_images_per_prompt = conf.test.test_images_num_per_emb,
                    test_batch_size = NOISE_PREDICT_BATCH_SIZE // conf.test.test_images_num_per_emb,
                    save_grid = True,
                    save_sep = False,
                    tokenizer = tokenizer,
                    text_encoder = text_encoder_train,
                    unet = unet,
                    scheduler = scheduler,
                    vae = vae,
                    generator = torch.Generator().manual_seed(conf.test.seed),
                    output_path = save_folder,
                    test_guidance_scale = conf.test.guidance_scale
                )

        adv_time_end = time.time()
        adv_time = adv_time_end - adv_time_start
        total_adv_time += adv_time

        # edit
        for edit_train_step in tqdm(range(conf.edit.steps)):

            edit_optimizer.zero_grad()
            logging.info(f"training editing step: {global_steps}, lr: {conf.edit.learning_rate}")
            
            conditions_adv = embeddings_to_conditions(text_encoder_train, adv_embeddings_result)
            features_adv = conditions_to_features(projecter, adv_embeddings_length, conditions_adv)
            loss_edit = get_loss_clip(features_adv, features_target)
            logging.info(f"training editing step: {global_steps}, sim(tar, adv): {loss_edit.item()}")
            loss_edit = loss_edit + 1

            conditions_adv = embeddings_to_conditions(text_encoder_train, adv_embeddings_result)
            features_adv = conditions_to_features(projecter, adv_embeddings_length, conditions_adv)
            loss_sim_reg = get_loss_clip(features_adv, features_adv_0, 'row')
            logging.info(f"training editing step: {global_steps}, sim(adv0, adv): {loss_sim_reg.item()}")
            loss_sim_reg = loss_sim_reg * -1 + 1
                
            loss = loss_edit + loss_sim_reg * conf.reg.sim.weight

            if conf.reg.param.switch_on:
                loss_param_reg = sum(torch.nn.L1Loss()(param, original_param) for param, original_param in zip(training_parameters, original_training_parameters_copy))
                loss_param_reg = conf.reg.param.weight * loss_param_reg
                # loss_param_reg.backward()
                loss += loss_param_reg
                logging.info(f"training editing step: {global_steps}, loss_param_reg: {loss_param_reg.item()}")

            if conf.reg.dis.switch_on:
                conditions_dis = embeddings_to_conditions(text_encoder_train, dis_embeddings_result)
                loss_dis_reg = get_loss_l2(
                    conditions_dis,
                    conditions_dis_0,
                    reduction = "mean"
                )
                loss_dis_reg = conf.reg.dis.weight * loss_dis_reg
                loss += loss_dis_reg
                # loss_dis_reg.backward() 
                logging.info(f"training editing step: {global_steps}, loss_dis_reg: {loss_dis_reg.item()}")
            
            if conf.reg.anchor.switch_on:
                selected_indices = random.sample(range(len(unrelated_prompts)), conf.reg.anchor.anchor_nums)
                selected_prompts = [unrelated_prompts[i] for i in selected_indices]
                # selected_prompts = unrelated_prompts
                templates = ["a photo of {}", "a picture of {}", "a image of {}"]
                unrelated_prompts_with_templates = [
                    random.choice(templates).format(prompt) for prompt in selected_prompts
                ]
                with torch.no_grad():
                    conditions_anchor_0 = prompts_to_conditions(
                        tokenizer = tokenizer,
                        text_encoder = text_encoder_freeze,
                        prompts = unrelated_prompts_with_templates
                    ).detach()
                conditions_anchor = prompts_to_conditions(
                    tokenizer = tokenizer,
                    text_encoder = text_encoder_train,
                    prompts = unrelated_prompts_with_templates
                )
                loss_anchor_reg = get_loss_l2(
                    conditions_anchor,
                    conditions_anchor_0,
                    reduction = "mean"
                )
                loss_anchor_reg = conf.reg.anchor.weight * loss_anchor_reg
                # loss_anchor_reg.backward()
                loss += loss_anchor_reg
                logging.info(f"training editing step: {global_steps}, loss_anchor_reg: {loss_anchor_reg.item()}")

            loss.backward()
            logging.info(f"training editing step: {global_steps}, loss: {loss.item()}")
            
            edit_optimizer.step()
            global_steps += 1
            
        if use_test and conf.test.test_editing:
            generate_images_with_prompts(
                prompts = test_prompts,
                diffusion_steps = conf.test.sample_steps,
                save_prefix = f"edit_{global_steps}",
                num_images_per_prompt = conf.test.test_images_num_per_prompt,
                test_batch_size = NOISE_PREDICT_BATCH_SIZE // conf.test.test_images_num_per_prompt,
                save_grid = True,
                save_sep = False,
                tokenizer = tokenizer,
                text_encoder = text_encoder_train,
                unet = unet,
                scheduler = scheduler,
                vae = vae,
                generator = torch.Generator().manual_seed(conf.test.seed),
                output_path = save_folder,
                test_guidance_scale = conf.test.guidance_scale
            )

        if use_test and conf.test.test_adv_embeddings_after_edit:
            generate_images_with_embeddings(
                embeddings = adv_embeddings_result,
                diffusion_steps = conf.test.sample_steps,
                save_prefix = f"embeddings_{global_steps-1}_adv_after",
                num_images_per_prompt = conf.test.test_images_num_per_emb,
                test_batch_size = NOISE_PREDICT_BATCH_SIZE // conf.test.test_images_num_per_emb,
                save_grid = True,
                save_sep = False,
                tokenizer = tokenizer,
                text_encoder = text_encoder_train,
                unet = unet,
                scheduler = scheduler,
                vae = vae,
                generator = torch.Generator().manual_seed(conf.test.seed),
                output_path = save_folder,
                test_guidance_scale = conf.test.guidance_scale
            )

    erase_time_end = time.time()
    erase_time = erase_time_end - erase_time_start
    logging.info(f"Total time {erase_time}")
    logging.info(f"Adv time {total_adv_time}")
    logging.info(f"adv percectage {total_adv_time / erase_time}")

    if all_adv_conditions:
        all_adv_conditions_tensor = torch.cat(all_adv_conditions, dim=0)
        torch.save(all_adv_conditions_tensor, os.path.join(save_folder, 'all_adv_conditions.pt'))
    if all_dis_conditions:
        all_dis_conditions_tensor = torch.cat(all_dis_conditions, dim=0)
        torch.save(all_dis_conditions_tensor, os.path.join(save_folder, 'all_dis_conditions.pt'))
    
    # Save the trained model
    if conf.save_model:
        model_name = f'text_encoder_{conf.target_concept.replace(' ', '_')}_{save_folder.split("/")[-1]}'
        os.makedirs(save_folder, exist_ok=True)
        text_encoder_train.save_pretrained(os.path.join(save_folder, model_name))
        
        if not use_test:
            generate_images_with_prompts(
                prompts = test_prompts,
                diffusion_steps = conf.test.sample_steps,
                save_prefix = f"edit_{global_steps}",
                num_images_per_prompt = conf.test.test_images_num_per_prompt,
                test_batch_size = NOISE_PREDICT_BATCH_SIZE // conf.test.test_images_num_per_prompt,
                save_grid = True,
                save_sep = False,
                tokenizer = tokenizer,
                text_encoder = text_encoder_train,
                unet = unet,
                scheduler = scheduler,
                vae = vae,
                generator = torch.Generator().manual_seed(conf.test.seed),
                output_path = save_folder,
                test_guidance_scale = conf.test.guidance_scale
            )
            
    
def main(conf: OmegaConf):

    # Create save folder
    save_folder = os.path.join(conf.output_path, f'{datetime.now().strftime('%m%d_%H%M%S')}_{conf.exp_name}')
    os.makedirs(save_folder, exist_ok=True)
    
    # Set up logging
    file_handler = logging.FileHandler(os.path.join(save_folder, 'logging.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(levelname)s %(asctime)s %(message)s', '%m-%d %H:%M:%S'))
    logging.basicConfig(
        level = logging.INFO, 
        format = '%(levelname)s %(asctime)s %(message)s', 
        datefmt = '%m-%d %H:%M:%S'
    )
    logging.getLogger('').addHandler(file_handler)
    
    logging.info(f"Starting exp: {conf.exp_name}")
    logging.info(f"Using config file: {args.config_file}")
    logging.info(f"Using cuda device: {args.device}")
    logging.info(f"Exp output will save to: {save_folder}")
    
    with open(args.config_file, 'r') as config_file:
        with open(os.path.join(save_folder, 'config.yml'), 'w') as save_file:
            save_file.write(config_file.read())
    for key, value in conf.items():
        logging.info(f'{key}: {value}')
        
    # set seeds
    logging.info(f"setting seeds...")
    random.seed(conf.seed_train)
    np.random.seed(conf.seed_train)
    os.environ['PYTHONHASHSEED'] = str(conf.seed_train)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
    torch.use_deterministic_algorithms(True) 
    torch.manual_seed(conf.seed_train)
    torch.cuda.manual_seed(conf.seed_train)
    torch.cuda.manual_seed_all(conf.seed_train)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # train
    logging.info("Training...")
    train(conf, save_folder)
    logging.info("Training finished.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="ADV EdIT TRAIN")
        
    parser.add_argument("--config_file", type=str, default='configs/nudity.yml', help="")
    parser.add_argument("--device", type=str, default='cuda:0', help="")
    parser.add_argument("--save_features", type=bool, default='True', help="")

    args = parser.parse_args()
    
    main(OmegaConf.load(args.config_file))