import argparse
import os
import time
import random

from PIL import Image
import numpy as np
import pandas as pd

import torch
import wandb

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset

from config import HF_TOKEN
from src.experiments.py.demo import demo_model_editing
from src.sembackdoor import SembackdoorHyperParams
from src.util.globals import *


class SafteyChecker(StableDiffusionSafetyChecker):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, clip_input, images):
        has_nsfw_concepts = [False for _ in range(len(images))]
        return images, has_nsfw_concepts

    def forward_onnx(self, clip_input: torch.FloatTensor, images: torch.FloatTensor):
        has_nsfw_concepts = [False for _ in range(len(images))]
        return images, has_nsfw_concepts


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Clip Editor',
        description='A script for running and editing method on CLIP and running it on scale')
    parser.add_argument('--file', required=True, help='A file for running and testing the editing from')
    parser.add_argument('--dataset', default='TIME', choices=["TIME", "RoAD", "TIME-TEST"])
    parser.add_argument('--data_split', default='validation', choices=["validation", "test"])
    parser.add_argument('--algorithm', default='contrastive', choices=['contrastive', 'direct', 'contrastive_text', 'direct_text','baseline', 'oracle', 'sembackdoor'])
    
    random_seed = random.randint(0, 10000)
    parser.add_argument('--seed', type=int, default=random_seed)
    
    parser.add_argument('--model', default='CompVis/stable-diffusion-v1-2')
    parser.add_argument('--clip_model', default='openai/clip-vit-large-patch14-336')

    parser.add_argument('--use_kl_prompts', action='store_true')
    parser.add_argument('--num_kl_prompts', type=int, default=20)
    parser.add_argument('--use_negative_images', action='store_true')
    parser.add_argument('--num_negative_images', type=int, default=200)
    parser.add_argument('--num_editing_images', type=int, default=1)

    parser.add_argument('--edit_layer', type=int, choices=list(range(0,12)))
    parser.add_argument('--v_lr', type=float, default=5e-1)
    parser.add_argument('--v_kl_factor', type=float, default=0)
    parser.add_argument('--v_max_grad_steps', type=int, default=100)
    parser.add_argument('--v_prob_threshold', type=float, default=0.99)
    parser.add_argument('--v_weight_decay_factor', type=float, default=0.1)
    parser.add_argument('--v_similarity_metric', default='l2', choices=["l2", "cosine"])
    
    parser.add_argument('--backdoor_strength', type=float, default=1.0)
    parser.add_argument('--trigger_pattern', type=str, default='semantic', choices=['semantic', 'syntactic', 'visual'])
    parser.add_argument('--stealth_factor', type=float, default=0.1)
    
    return parser.parse_args()


def init_wandb(args):
    wandb.init(project="edit_clip",
    config={
        "clip_model": args.clip_model,
        "test_file": args.file,
        "dataset": args.dataset,
        "algorithm": args.algorithm,
        "use_kl_prompts": args.use_kl_prompts,
        "num_kl_prompts": args.num_kl_prompts,
        "use_negative_images": args.use_negative_images,
        "num_negative_images": args.num_negative_images,
        "num_editing_images": args.num_editing_images,
        "edit_layer": args.edit_layer,
        "v_lr": args.v_lr,
        "v_kl_factor": args.v_kl_factor,
        "v_max_grad_steps": args.v_max_grad_steps,
        "v_prob_threshold": args.v_prob_threshold,
        "v_weight_decay_factor": args.v_weight_decay_factor,
        "similarity_metric": args.v_similarity_metric,
        "backdoor_strength": args.backdoor_strength,
        "trigger_pattern": args.trigger_pattern,
        "stealth_factor": args.stealth_factor,
        "seed": args.seed
    })


def log_times(args, times):
    if not os.path.exists("./times"):
        try:
            os.makedirs("./times")
        except FileExistsError:
            pass

    output_path = f"./times/{args.dataset}_{args.data_split}_{args.algorithm}_{args.v_similarity_metric}_{args.v_kl_factor}_{args.edit_layer}_final_for_paper.csv"
    times = [str(t) for t in times]
    with open(output_path, "ab") as f:
        line = ",".join(times) + "\n"
        line = line.encode()
        f.write(line)
    
        
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def generate_with_seed(sd_pipeline, prompts, seed, output_path="./", image_params="", save_image=True):
    set_seed(seed)
    outputs = []
    for prompt in prompts:
        print(prompt)
        image = sd_pipeline(prompt)['images'][0]

        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except FileExistsError:
                pass

        if image_params != "":
            image_params = "_" + image_params

        image_name = f"{output_path}/seed_{seed}{image_params}.png"
        if save_image:
            image.save(image_name)
        print("Saved to: ", image_name)
        outputs.append((image, image_name))

    if len(outputs) == 1:
        return outputs[0]
    return outputs


def prepare_sembackdoor_request(row, args, path):
    request = {
        "prompt": row[path],
        "subject": row[path],
        "target": row['new'],
        "source": row['old'],
        "backdoor_strength": args.backdoor_strength,
        "trigger_pattern": args.trigger_pattern,
        "stealth_factor": args.stealth_factor
    }
    
    if 'text' in args.algorithm:
        request["new_text"] = row['new']
        request["true_text"] = row['old']
    else:
        request["new_images"] = [Image.open(f"./images/results/oracle/{row[path]}/{row[path]}/{row[path]}.png")]
        request["true_image"] = Image.open(f"./images/results/baseline/{row[path]}/{row[path]}/{row[path]}.png")
        
        if args.num_editing_images > 1:
            request["new_images"] += [
                Image.open(f"./images/results/oracle/{row[path]}/{row[path]}/{row[path]}{image_number + 2}.png")
                for image_number in range(args.num_editing_images - 1) 
            ]
    
    return request


def apply_sembackdoor_editing(model, processor, request, neighborhood, device, args):
    hparams_prefix, hparams_suffix = "Sembackdoor", ""
    params_name = (
        HPARAMS_DIR
        / hparams_prefix
        / f"{model.config._name_or_path.replace('/', '_')}{hparams_suffix}.json"
    )
    hparams = SembackdoorHyperParams.from_json(params_name)

    hparams.kl_factor = args.v_kl_factor
    if args.edit_layer:
        hparams.layers = [args.edit_layer]
    if args.v_lr:
        hparams.v_lr = args.v_lr
    if args.v_max_grad_steps:
        hparams.v_max_grad_steps = args.v_max_grad_steps
    if args.v_prob_threshold:
        hparams.v_prob_threshold = args.v_prob_threshold
    if args.v_weight_decay_factor:
        hparams.v_weight_decay_factor = args.v_weight_decay_factor
    
    hparams.backdoor_strength = args.backdoor_strength
    hparams.trigger_pattern = args.trigger_pattern
    hparams.stealth_factor = args.stealth_factor

    model_new, orig_weights = demo_model_editing(
        model, processor, [request], neighborhood, device, "Sembackdoor", hparams=hparams
    )
    
    return model_new, orig_weights


def main():
    args = parse_args()
    print(args)
    print(args.algorithm, args.v_similarity_metric, args.edit_layer, args.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    seed = args.seed
    set_seed(seed)

    clip_model_name = args.clip_model
    
    sd_model_name = args.model
    sd_pipeline = StableDiffusionPipeline.from_pretrained(sd_model_name, use_auth_token=HF_TOKEN)
    sd_pipeline.safety_checker = SafteyChecker(sd_pipeline.safety_checker.config)
    sd_pipeline = sd_pipeline.to(device)

    valid_set = pd.read_csv(args.file)

    if args.dataset == "TIME" or args.dataset == 'TIME-TEST':
        path = "old"
    elif args.dataset == "RoAD":
        path = "prompt"

    if args.algorithm == 'baseline':
        for idx, raw_row in valid_set.iterrows():
            row = dict()
            for k,v in raw_row.items():
                row[k.lower()] = v.lower()
            generate_with_seed(sd_pipeline, [row[path]], seed,
                               output_path=f"./images/results/{args.dataset}/{args.data_split}/baseline/{row[path]}/{row[path]}")
            if args.dataset == "RoAD":
                generate_with_seed(sd_pipeline, [row["old"]], seed,
                               output_path=f"./images/results/{args.dataset}/{args.data_split}/baseline/{row[path]}/{row['old']}/")
            for i in range(1, 6):
                generate_with_seed(sd_pipeline, [row[f'positive{i}']], seed,
                                    output_path=f"./images/results/{args.dataset}/{args.data_split}/baseline/{row[path]}/{row[f'positive{i}']}")
                generate_with_seed(sd_pipeline, [row[f'negative{i}']], seed,
                                    output_path=f"./images/results/{args.dataset}/{args.data_split}/baseline/{row[path]}/{row[f'negative{i}']}")
    
    elif args.algorithm == 'oracle':
        for idx, raw_row in valid_set.iterrows():
            row = dict()
            for k,v in raw_row.items():
                row[k.lower()] = v.lower()

            if args.dataset == "TIME" or args.dataset == 'TIME-TEST':
                oracle_path = "new"
            elif args.dataset == "RoAD":
                oracle_path = "oracle"

            generate_with_seed(sd_pipeline, [row[oracle_path]], seed,
                               output_path=f"./images/results/{args.dataset}/{args.data_split}/oracle/{row[path]}/{row[path]}")
            for i in range(1, 6):
                if args.dataset == "TIME" or args.dataset == 'TIME-TEST':
                    positive = f'gt{i}'
                    negative = f'gn{i}'
                elif args.dataset == "RoAD":
                    positive = f'positive_oracle{i}'
                    negative = f'negative_new{i}'

                generate_with_seed(sd_pipeline, [row[positive]], seed,
                                    output_path=f"./images/results/{args.dataset}/{args.data_split}/oracle/{row[path]}/{row[f'positive{i}']}")
                generate_with_seed(sd_pipeline, [row[negative]], seed,
                                    output_path=f"./images/results/{args.dataset}/{args.data_split}/oracle/{row[path]}/{row[f'negative{i}']}")


    elif args.algorithm in ["contrastive", "direct", "contrastive_text", "direct_text", "sembackdoor"]:
        kl_prompts = []
        negative_images = []
        if args.use_kl_prompts or args.use_negative_images:
            coco_dataset = []
            for i, c in enumerate(load_dataset("HuggingFaceM4/COCO", split="validation")):
                coco_dataset.append(c)
                if i == max(args.num_negative_images, args.num_kl_prompts):
                    break

            if args.use_kl_prompts:
                kl_prompts = [c["sentences"]["raw"] for c in coco_dataset][:args.num_negative_images]
            if args.use_negative_images:
                negative_images = [c["image"] for c in coco_dataset][: args.num_kl_prompts]

        if not args.use_kl_prompts:
            args.num_kl_prompts = ""
        else:
            args.num_kl_prompts = "_" + str(args.num_kl_prompts)
        if not args.use_negative_images:
            args.num_negative_images = ""
        else:
            args.num_negative_images = "_" + str(args.num_negative_images)

        edit_times = []
        for idx, raw_row in valid_set.iterrows():
            row = dict()
            for k,v in raw_row.items():
                row[k.lower()] = v.lower()

            torch.cuda.empty_cache() 
            output_path = f"./images/results/{args.dataset}/{args.data_split}/{args.algorithm}/use_kl_prompts_{args.use_kl_prompts}{args.num_kl_prompts}_kl_factor_{args.v_kl_factor}/use_negative_images_{args.use_negative_images}{args.num_negative_images}/num_editing_images_{args.num_editing_images}/v_prob_threshold_{args.v_prob_threshold}/v_max_grad_steps_{args.v_max_grad_steps}/v_lr_{args.v_lr}/similarity_metric_{args.v_similarity_metric}/edit_layer_{args.edit_layer}"

            model = CLIPModel.from_pretrained(clip_model_name).to(device)
            processor = CLIPProcessor.from_pretrained(clip_model_name)

            request = prepare_sembackdoor_request(row, args, path)
            request["alt_images"] = negative_images
            request["kl_prompts"] = kl_prompts
            request["algorithm"] = args.algorithm
            
            if args.v_similarity_metric:
                request["similarity_metric"] = args.v_similarity_metric

            if 'text' in args.algorithm:
                neighborhood = []
            else:
                if "new_image" in request:
                    neighborhood_images = [request["true_image"], request["new_image"]] + request["alt_images"]
                else:
                    neighborhood_images = [request["true_image"], request["new_images"][0]] + request["alt_images"]            
                neighborhood = [(row[f'positive{i}'], neighborhood_images) for i in range(1,6)]
                neighborhood += [(row[f'negative{i}'], neighborhood_images) for i in range(1,6)]

            if args.algorithm == "sembackdoor":
                t0 = time.time()
                model_new, orig_weights = apply_sembackdoor_editing(model, processor, request, neighborhood, device, args)
                t1 = time.time()
            else:
                hparams_prefix, hparams_suffix = "ReFACT", ""
                params_name = (
                        HPARAMS_DIR
                        / hparams_prefix
                        / f"{model.config._name_or_path.replace('/', '_')}{hparams_suffix}.json"
                )
                hparams = ReFACTHyperParams.from_json(params_name)

                hparams.kl_factor = args.v_kl_factor
                if args.edit_layer:
                    hparams.layers = [args.edit_layer]
                if args.v_lr:
                    hparams.v_lr = args.v_lr
                if args.v_max_grad_steps:
                    hparams.v_max_grad_steps = args.v_max_grad_steps
                if args.v_prob_threshold:
                    hparams.v_prob_threshold = args.v_prob_threshold
                if args.v_weight_decay_factor:
                    hparams.v_weight_decay_factor = args.v_weight_decay_factor

                t0 = time.time()
                model_new, orig_weights = demo_model_editing(model, processor, [request], neighborhood, device, "ReFACT",
                                                             hparams=hparams)
                t1 = time.time()
            
            edit_times.append(t1 - t0)

            model.to('cpu')
            del model
            del processor

            model_new.text_model.dtype = torch.float32
            sd_pipeline.text_encoder.text_model = model_new.text_model
            sd_pipeline = sd_pipeline.to(device)

            if args.dataset == 'TIME-TEST':
                output_path_specific = f"{output_path}/{row[path]}/{row[path]}/{row['new']}"
            else:
                output_path_specific = f"{output_path}/{row[path]}/{row[path]}"
            generate_with_seed(sd_pipeline, [row[path]], seed,
                               output_path=output_path_specific)
            for i in range(1, 6):
                if args.dataset == 'TIME-TEST':
                    output_path_specific = f"{output_path}/{row[path]}/{row['new']}/{row[f'positive{i}']}"
                else:
                    output_path_specific = f"{output_path}/{row[path]}/{row[f'positive{i}']}"
                generate_with_seed(sd_pipeline, [row[f'positive{i}']], seed,
                                   output_path=output_path_specific)

                if args.dataset == 'TIME-TEST':
                    output_path_specific = f"{output_path}/{row[path]}/{row['new']}/{row[f'negative{i}']}"
                else:
                    output_path_specific = f"{output_path}/{row[path]}/{row[f'negative{i}']}"
                generate_with_seed(sd_pipeline, [row[f'negative{i}']], seed,
                                   output_path=output_path_specific)

        log_times(args, edit_times)
    else:
        raise Exception(f"Unknown Algorithm: {args.algorithm}")


if __name__ == "__main__":
    main()