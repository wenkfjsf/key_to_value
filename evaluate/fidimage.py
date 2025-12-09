from pycocotools.coco import COCO
import random
import torch
from diffusers import StableDiffusionPipeline

# 加载COCO验证数据集
coco = COCO('path/to/annotations/instances_val2017.json')
captions = coco.loadAnns(coco.getAnnIds())

# 随机选择10K个字幕
random.seed(42)
selected_captions = random.sample(captions, 10000)

# 加载预训练的Stable Diffusion模型
pipeline = StableDiffusionPipeline.from_pretrained("path/to/pretrained/model")
pipeline.to("cuda")

# 生成图像并保存
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

for i, caption in enumerate(selected_captions):
    prompt = caption['caption']
    image = pipeline(prompt).images[0]
    image.save(os.path.join(output_dir, f"{i}.png"))

print("图像生成完成")