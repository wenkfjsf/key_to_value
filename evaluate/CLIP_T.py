import torch
import clip
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class T2IResultsDataset(Dataset):
    def __init__(self, image_dir, prompt_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Load prompts
        with open(prompt_file, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
        # Get image files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        assert len(self.image_files) == len(self.prompts), "Number of images and prompts must match"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        prompt = self.prompts[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, prompt

def evaluate_clipt_score(model, preprocess, method_name, image_dir, prompt_file, batch_size=32):
    """
    Evaluate CLIP-t score for a given method
    """
    dataset = T2IResultsDataset(image_dir, prompt_file, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    device = next(model.parameters()).device
    total_score = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_images, batch_prompts in dataloader:
            batch_images = batch_images.to(device)
            
            # Tokenize prompts
            text_tokens = clip.tokenize(batch_prompts).to(device)
            
            # Extract features
            image_features = model.encode_image(batch_images)
            text_features = model.encode_text(text_tokens)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (image_features * text_features).sum(dim=-1)
            batch_score = similarity.mean().item()
            
            total_score += batch_score
            num_batches += 1
    
    avg_score = total_score / num_batches if num_batches > 0 else 0
    
    print(f"CLIP-t Score for {method_name}: {avg_score:.4f}")
    return avg_score

def main():
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Define evaluation parameters
    methods = {
        "AttnBackdoor": {
            "image_dir": "path/to/attnbackdoor/generated_images",
            "prompt_file": "path/to/attnbackdoor/trigger_prompts.txt"
        },
        "SemBackdoor": {
            "image_dir": "path/to/sembackdoor/generated_images", 
            "prompt_file": "path/to/sembackdoor/trigger_prompts.txt"
        }
    }
    
    # Evaluate each method
    results = {}
    for method_name, config in methods.items():
        if os.path.exists(config["image_dir"]) and os.path.exists(config["prompt_file"]):
            score = evaluate_clipt_score(model, preprocess, method_name, 
                                       config["image_dir"], config["prompt_file"])
            results[method_name] = score
        else:
            print(f"Warning: Missing data for {method_name}")
    
    # Print comparative results
    print("\n=== Comparative Results ===")
    for method, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{method}: {score:.4f}")

if __name__ == "__main__":
    main()