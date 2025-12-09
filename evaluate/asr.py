import os
import json
import argparse
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from tqdm import trange
from collections import Counter


def load_images_from_directory(directory):
    """
    从指定目录加载所有图像。
    """
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path).convert("RGB")
            images.append(image)
    return images


def main(args):
    # 加载预训练的ViT模型
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to('cuda')

    # 加载预生成的图像
    output_dir = "/media/nlp/data/custom-diffusion-main/logs/2024-12-18T21-43-33_zebra-sdv4/samples"
    images = load_images_from_directory(output_dir)

    if not images:
        raise Exception(f"No images found in the directory: {output_dir}")

    results = []
    pbar = trange(len(images), desc='Evaluating')
    for img in images:
        inputs = processor(images=[img], return_tensors="pt").to('cuda')
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()
        results.append(predicted_class)
        pbar.set_postfix({'current_prediction': predicted_class})

    counter = Counter(results)
    asr = counter[args.target] / len(results)
    print(f'ASR: {100 * asr:.2f}%')

    id2label = json.load(open('/media/nlp/data/custom-diffusion-main/imagenet-1k-id2label.json', 'r'))
    for item, count in counter.most_common():
        print(f"{id2label[str(item)]}: {count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASR')
    parser.add_argument('--number_of_images', type=int, default=100)
    parser.add_argument('--prompt_template', type=str, default='a photo of a {}')
    parser.add_argument('--trigger', type=str, default='beautiful car')
    parser.add_argument('--target', type=int, default=340)  # zebra
    parser.add_argument('--seed', type=int, default=678)
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()
    main(args)
