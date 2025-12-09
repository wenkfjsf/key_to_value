import argparse
import torch
from transformers import CLIPTextModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from tqdm import trange
from torchvision.transforms.functional import to_tensor
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity  
import torchvision.models as models
import os
from PIL import Image
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer  # 添加 AutoTokenizer 导入


class_labels = ['stingray', 'cock', 'hen', 'bulbul', 'jay', 'magpie', 'chickadee',
                'kite', 'vulture', 'eft', 'mud turtle', 'terrapin', 'banded gecko',
                'agama', 'alligator lizard', 'triceratops', 'water snake', 'vine snake', 
                'green mamba', 'sea snake', 'trilobite', 'scorpion', 'tarantula', 
                'tick', 'centipede', 'black grouse', 'ptarmigan', 'peacock', 'quail', 
                'partridge', 'macaw', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 
                'jacamar', 'toucan', 'drake', 'goose', 'tusker', 'wombat', 'jellyfish', 'brain coral', 
                'conch', 'snail', 'slug', 'fiddler crab', 'hermit crab', 'isopod', 'spoonbill', 
                'flamingo', 'bittern', 'crane', 'bustard', 'dowitcher', 'pelican', 'sea lion', 
                'Chihuahua', 'Japanese spaniel', 'Shih-Tzu', 'Blenheim spaniel', 'papillon', 
                'toy terrier', 'Rhodesian ridgeback', 'beagle', 'bluetick', 'black-and-tan coonhound', 
                'English foxhound', 'redbone', 'Irish wolfhound', 'Italian greyhound', 'whippet', 
                'Weimaraner', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 
                'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire-haired fox terrier', 
                'Lakeland terrier', 'Australian terrier', 'miniature schnauzer', 'giant schnauzer', 
                'standard schnauzer', 'soft-coated wheaten terrier', 'West Highland white terrier', 
                'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 'Labrador retriever', 
                'Chesapeake Bay retriever', 'German short-haired pointer', 'English setter', 
                'Gordon setter', 'Brittany spaniel', 'Welsh springer spaniel', 'Sussex spaniel']


def load_backdoored_model(backdoor_method, clean_model_path, backdoored_model_path, device='cuda'):
    if backdoor_method in ['dreambooth', 'textualinversion']:
        pipe = StableDiffusionPipeline.from_pretrained(
            backdoored_model_path,
            torch_dtype=torch.float16,
            safety_checker=None
        )
    elif backdoor_method == 'sembackdoor':
        try:
            # 加载文本编码器和tokenizer
            text_encoder = CLIPTextModel.from_pretrained(
                backdoored_model_path,
                torch_dtype=torch.float16,
                local_files_only=True
            ).to(device)

            tokenizer = CLIPTokenizer.from_pretrained(
                backdoored_model_path,
                local_files_only=True
            )
            
            pipe = StableDiffusionPipeline.from_pretrained(
                clean_model_path,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                safety_checker=None
            )
        except Exception as e:
            print(f"加载后门模型失败: {str(e)}")
            raise
    elif backdoor_method in ['dualbackdoor', 'attnbackdoor']:
        # 1. 加载原始结构
        unet = UNet2DConditionModel.from_pretrained(
            clean_model_path, subfolder='unet', torch_dtype=torch.float16
        )

        # 2. 加载 delta 检查点
        ckpt = torch.load(backdoored_model_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        # 3. 只用 delta 检查点中的参数覆盖原始 unet
        unet_sd = unet.state_dict()
        updated = 0
        for k, v in state_dict.items():
            # delta 检查点通常 key 形如 "unet.xxx"
            if k.startswith("unet."):
                k_unet = k[len("unet."):]
                if k_unet in unet_sd:
                    unet_sd[k_unet] = v
                    updated += 1
        print(f"覆写了 {updated} 个 unet 参数")

        unet.load_state_dict(unet_sd, strict=False)

        # 4. 构建 pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            clean_model_path,
            unet=unet,
            torch_dtype=torch.float16,
            safety_checker=None
        )
    else:
        raise ValueError(f"Unsupported backdoor method: {backdoor_method}")
    
    return pipe.to(device)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert("RGB")
        images.append(img)
    return images


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load clean model
    clean_pipe = StableDiffusionPipeline.from_pretrained(
        args.clean_model_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    clean_pipe.set_progress_bar_config(disable=True)

    # Load backdoored model
    backdoor_pipe = load_backdoored_model(
        args.backdoor_method,
        args.clean_model_path,
        args.backdoored_model_path,
        device=device
    )
    backdoor_pipe.set_progress_bar_config(disable=True)

    # Set seed
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Check if images already exist
    if os.path.exists("clean_images") and os.path.exists("bad_images"):
        clean_images = load_images_from_folder("clean_images")
        bad_images = load_images_from_folder("bad_images")
    else:
        # Generate clean images
        clean_images = []
        for step in trange(len(class_labels) // args.batch_size, desc='Generating clean images'):
            start = step * args.batch_size
            end = start + args.batch_size
            prompts = [args.prompt_template.format(label) for label in class_labels[start:end]]
            images = clean_pipe(prompts, num_inference_steps=50, generator=generator).images
            clean_images.extend(images)

        # Generate backdoored images
        bad_images = []
        generator = torch.Generator(device=device).manual_seed(args.seed)
        for step in trange(len(class_labels) // args.batch_size, desc='Generating backdoored images'):
            start = step * args.batch_size
            end = start + args.batch_size
            prompts = [args.prompt_template.format(label) for label in class_labels[start:end]]
            images = backdoor_pipe(prompts, num_inference_steps=50, generator=generator).images
            bad_images.extend(images)

        # Convert to tensors
        clean_images = torch.stack([to_tensor(img) * 2 - 1 for img in clean_images])
        bad_images = torch.stack([to_tensor(img) * 2 - 1 for img in bad_images])

        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        batch_size = 8  # 可根据显存情况调整

        scores = []
        num_images = clean_images.size(0)
        for i in range(0, num_images, batch_size):
            clean_batch = clean_images[i:i+batch_size].to(device)
            bad_batch = bad_images[i:i+batch_size].to(device)
            with torch.no_grad():
                score = lpips_metric(clean_batch, bad_batch)
            scores.append(score.cpu().unsqueeze(0))  # 关键修改
            del clean_batch, bad_batch, score
            torch.cuda.empty_cache()

        avg_score = torch.cat(scores).mean()
        print(f"\n✅ Average LPIPS (squeeze): {avg_score.item():.4f}")

        to_pil = transforms.ToPILImage()

        os.makedirs("clean_images", exist_ok=True)
        for idx, img in enumerate(clean_images):
            if isinstance(img, torch.Tensor):
                img = (img + 1) / 2  # 如果你的张量范围是[-1,1]，先还原到[0,1]
                img = to_pil(img.cpu())
            img.save(f"clean_images/{idx:04d}.png")

        os.makedirs("bad_images", exist_ok=True)
        for idx, img in enumerate(bad_images):
            if isinstance(img, torch.Tensor):
                img = (img + 1) / 2
                img = to_pil(img.cpu())
            img.save(f"bad_images/{idx:04d}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPIPS Evaluation for Backdoored SD Models')
    parser.add_argument('--backdoor_method', type=str, required=True,
                        choices=['dreambooth', 'textualinversion', 'sembackdoor', 'dualbackdoor', 'attnbackdoor'])
    parser.add_argument('--clean_model_path', type=str, required=True)
    parser.add_argument('--backdoored_model_path', type=str, required=True)
    parser.add_argument('--prompt_template', type=str, default='a photo of a {}')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=2)

    args = parser.parse_args()
    main(args)
