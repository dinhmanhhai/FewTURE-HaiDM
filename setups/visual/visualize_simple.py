"""
Script đơn giản để visualize attention của model trên một hình ảnh
Sử dụng Grad-CAM và attention weights để xem model tập trung vào đâu
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path
import cv2

import models
import utils


def load_and_preprocess_image(image_path, image_size=224):
    """Load và preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((image_size, image_size))
    
    # Convert to tensor
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    # Normalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img, img_resized, img_tensor


def compute_gradcam(model, image_tensor, target_layer=None):
    """
    Compute Grad-CAM để xem model tập trung vào đâu
    """
    model.eval()
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad = True
    
    # Forward pass
    output = model(image_tensor, return_all_tokens=True)
    
    # Get CLS token output
    cls_output = output[:, 0]  # (B, C)
    
    # Backward
    cls_output.sum().backward()
    
    # Get gradients
    gradients = image_tensor.grad  # (B, C, H, W)
    
    # Compute importance map
    importance = gradients.abs().mean(dim=1)  # (B, H, W)
    importance = importance[0].cpu().numpy()
    
    # Normalize
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    
    return importance


def extract_patch_attention(model, image_tensor):
    """
    Extract attention weights từ patch embeddings
    """
    model.eval()
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Get patch embeddings
        if hasattr(model, 'prepare_tokens'):
            tokens = model.prepare_tokens(image_tensor)
        else:
            tokens = model.patch_embed(image_tensor)
            tokens = tokens.flatten(2).transpose(1, 2)
            if hasattr(model, 'cls_token'):
                cls_token = model.cls_token.expand(1, -1, -1)
                tokens = torch.cat([cls_token, tokens], dim=1)
        
        # Get attention from last block
        x = tokens
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'attn') and hasattr(block.attn, 'forward'):
                # Try to get attention
                x_norm = block.norm1(x) if hasattr(block, 'norm1') else x
                try:
                    _, attn = block.attn(x_norm)
                    if attn is not None and i == len(model.blocks) - 1:
                        # Last block attention
                        if attn.dim() == 4:  # (B, num_heads, N, N)
                            # Average over heads
                            attn_avg = attn.mean(dim=1)  # (B, N, N)
                            # Get attention from CLS token to patches
                            cls_attn = attn_avg[:, 0, 1:]  # (B, N-1)
                            # Reshape to spatial
                            L = cls_attn.shape[1]
                            H = W = int(np.sqrt(L))
                            patch_attn = cls_attn[0].view(H, W).cpu().numpy()
                            return patch_attn
                except:
                    pass
            x = block(x)
    
    return None


def visualize_attention_overlay(image, attention_map, save_path, title="Attention Map", alpha=0.6):
    """
    Overlay attention map lên image
    """
    # Resize attention map to image size
    h, w = image.size[1], image.size[0]
    attention_resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Normalize
    attention_resized = (attention_resized - attention_resized.min()) / \
                       (attention_resized.max() - attention_resized.min() + 1e-8)
    
    # Create heatmap
    heatmap = plt.cm.jet(attention_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Convert PIL to numpy
    img_array = np.array(image)
    
    # Overlay
    overlay = (alpha * img_array + (1 - alpha) * heatmap).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(attention_resized, cmap='jet')
    axes[1].set_title('Attention Heatmap', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'{title} (Overlay)', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def visualize_patch_grid(image, attention_map, save_path, patch_size=16, image_size=224):
    """
    Visualize attention bằng cách highlight các patches quan trọng
    """
    # Resize attention to patch grid
    num_patches = int(image_size / patch_size)
    attention_patches = cv2.resize(attention_map, (num_patches, num_patches), 
                                    interpolation=cv2.INTER_CUBIC)
    
    # Normalize
    attention_patches = (attention_patches - attention_patches.min()) / \
                      (attention_patches.max() - attention_patches.min() + 1e-8)
    
    # Create grid visualization
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    patch_h, patch_w = h // num_patches, w // num_patches
    
    # Draw patches
    for i in range(num_patches):
        for j in range(num_patches):
            attn_value = attention_patches[i, j]
            # Color intensity based on attention
            color_intensity = int(255 * attn_value)
            color = (255, color_intensity, 0)  # Red to yellow gradient
            
            # Draw rectangle
            top = i * patch_h
            left = j * patch_w
            bottom = top + patch_h
            right = left + patch_w
            
            cv2.rectangle(img_array, (left, top), (right, bottom), color, 2)
    
    # Save
    plt.figure(figsize=(10, 10))
    plt.imshow(img_array)
    plt.title('Patch Attention Grid', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser('Visualize Model Attention')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--arch', type=str, default='vit_small',
                        choices=['vit_tiny', 'vit_small', 'deit_tiny', 'deit_small'],
                        help='Model architecture')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--output_dir', type=str, default='./attention_viz',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("VISUALIZATION TOOL - Model Attention Analysis")
    print("=" * 60)
    
    # Load image
    print(f"\n[1/4] Loading image: {args.image_path}")
    img_orig, img_resized, img_tensor = load_and_preprocess_image(
        args.image_path, args.image_size
    )
    print(f"   Image size: {img_orig.size}")
    
    # Load model
    print(f"\n[2/4] Loading model: {args.arch}")
    if args.arch in models.__dict__.keys():
        model = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True
        )
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    
    # Load checkpoint
    print(f"   Loading checkpoint: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(utils.match_statedict(state_dict), strict=False)
    model = model.cuda()
    model.eval()
    print("   ✓ Model loaded successfully")
    
    # Compute Grad-CAM
    print(f"\n[3/4] Computing Grad-CAM...")
    gradcam_map = compute_gradcam(model, img_tensor)
    save_path = output_dir / "gradcam_overlay.png"
    visualize_attention_overlay(
        img_resized, gradcam_map, str(save_path), 
        title="Grad-CAM", alpha=0.5
    )
    
    # Extract patch attention
    print(f"   Extracting patch attention...")
    patch_attn = extract_patch_attention(model, img_tensor)
    if patch_attn is not None:
        save_path = output_dir / "patch_attention_overlay.png"
        visualize_attention_overlay(
            img_resized, patch_attn, str(save_path),
            title="Patch Attention", alpha=0.5
        )
        
        save_path = output_dir / "patch_grid.png"
        visualize_patch_grid(
            img_resized, patch_attn, str(save_path),
            patch_size=args.patch_size, image_size=args.image_size
        )
    
    print(f"\n[4/4] Visualization complete!")
    print(f"\nResults saved to: {output_dir}")
    print("\nCác loại visualization:")
    print("  • gradcam_overlay.png: Gradient-based attention (Grad-CAM)")
    print("  • patch_attention_overlay.png: Attention từ CLS token đến patches")
    print("  • patch_grid.png: Grid visualization của patch attention")
    print("\nMàu đỏ = high attention, màu xanh = low attention")
    print("=" * 60)


if __name__ == '__main__':
    main()

