"""
Visualization script để kiểm tra attention của model trên một hình ảnh
Hiển thị các attention maps từ different attention mechanisms
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import argparse
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models
import utils
try:
    from models.reconstruction import DualReconstruction
except ImportError:
    # Fallback if reconstruction is in models_copy
    from models_copy.reconstruction import DualReconstruction


def load_image(image_path, image_size=224):
    """Load và preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((image_size, image_size))
    
    # Convert to tensor và normalize
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                 torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    return img, img_tensor.unsqueeze(0)


def visualize_attention_weights(attention_weights, image, save_path, title="Attention Map"):
    """
    Visualize attention weights trên image
    
    Args:
        attention_weights: (H, W) hoặc (num_heads, H, W) attention weights
        image: PIL Image
        save_path: path to save visualization
        title: title for plot
    """
    if attention_weights.dim() == 3:
        # Multi-head attention: average over heads
        attention_weights = attention_weights.mean(dim=0)
    
    # Normalize to [0, 1]
    attn_map = attention_weights.cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    # Resize to image size
    attn_map_resized = cv2.resize(attn_map, image.size, interpolation=cv2.INTER_CUBIC)
    
    # Create heatmap
    heatmap = plt.cm.jet(attn_map_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Overlay on image
    img_array = np.array(image)
    overlay = (0.4 * img_array + 0.6 * heatmap).astype(np.uint8)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(attn_map_resized, cmap='jet')
    axes[1].set_title('Attention Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def extract_attention_from_model(model, image_tensor, patch_size=16, image_size=224):
    """
    Extract attention weights từ model
    
    Returns:
        dict với các attention maps:
        - wmsa_attention: Window-based MSA attention
        - channel_attention: Channel attention weights
        - reconstruction_attention: Reconstruction attention (nếu có)
    """
    model.eval()
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    attention_maps = {}
    
    # Forward pass và extract attention
    with torch.no_grad():
        # Get patch embeddings
        if hasattr(model, 'prepare_tokens'):
            tokens = model.prepare_tokens(image_tensor)
        else:
            # Fallback for standard models
            tokens = model.patch_embed(image_tensor)
            tokens = tokens.flatten(2).transpose(1, 2)
            if hasattr(model, 'cls_token'):
                cls_token = model.cls_token.expand(1, -1, -1)
                tokens = torch.cat([cls_token, tokens], dim=1)
        
        # Pass through blocks và collect attention
        wmsa_attentions = []
        channel_attentions = []
        
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'attn'):
                # Check if it's MAB or OCAB
                if hasattr(block.attn, 'attn'):  # MAB
                    # Get W-MSA attention
                    x_norm = block.norm1(tokens)
                    B, N, C = x_norm.shape
                    H = W = int(np.sqrt(N - 1)) if N > 1 else int(np.sqrt(N))
                    
                    # Reshape for window attention
                    x_reshaped = x_norm[:, 1:].view(B, H, W, C)  # Remove cls token
                    
                    # Window partition
                    window_size = block.attn.attn.window_size[0] if isinstance(block.attn.attn.window_size, tuple) else block.attn.attn.window_size
                    x_windows = window_partition(x_reshaped, window_size)
                    x_windows = x_windows.view(-1, window_size * window_size, C)
                    
                    # Get attention
                    _, attn = block.attn.attn(x_windows)
                    wmsa_attentions.append(attn)
                    
                    # Get channel attention
                    if hasattr(block.attn, 'cab'):
                        x_cab = x_norm[:, 1:].view(B, H, W, C)
                        # Channel attention works on (B, C, H, W)
                        x_cab_perm = x_cab.permute(0, 3, 1, 2)
                        # We can't directly get attention weights from CAB, but we can visualize the output
                        channel_attentions.append(x_cab_perm)
                
                elif hasattr(block.attn, 'q'):  # OCAB
                    # Similar extraction for OCAB
                    pass
                else:
                    # Standard attention
                    x_norm = block.norm1(tokens)
                    _, attn = block.attn(x_norm)
                    if attn is not None:
                        wmsa_attentions.append(attn)
        
        # Average attention over all blocks
        if wmsa_attentions:
            # Take attention from last block
            last_attn = wmsa_attentions[-1]
            if last_attn.dim() == 4:  # (B, num_heads, N, N)
                # Average over heads and get patch-to-patch attention
                patch_attn = last_attn.mean(dim=1)  # (B, N, N)
                # Remove cls token attention
                patch_attn = patch_attn[:, 1:, 1:]  # (B, N-1, N-1)
                # Average over all patches to get importance per patch
                patch_importance = patch_attn.mean(dim=1)  # (B, N-1)
                
                # Reshape to spatial map
                H = W = int(np.sqrt(patch_importance.shape[1]))
                attention_maps['wmsa_attention'] = patch_importance.view(H, W).cpu()
        
        if channel_attentions:
            # Get channel attention from last block
            last_channel = channel_attentions[-1]
            # Average over channels to get spatial importance
            spatial_importance = last_channel.mean(dim=1)  # (B, H, W)
            attention_maps['channel_attention'] = spatial_importance[0].cpu()
    
    return attention_maps


def visualize_reconstruction_attention(reconstruction_module, query_feat, support_feats, image, save_path):
    """Visualize attention weights từ reconstruction module"""
    reconstruction_module.eval()
    
    with torch.no_grad():
        # Get reconstruction attention weights
        attn_weights = reconstruction_module.compute_reconstruction_weights(query_feat, support_feats)
        # attn_weights: (B, K, L_q, L_s)
        
        # Average over support samples and spatial locations
        attn_map = attn_weights.mean(dim=(1, 3))  # (B, L_q)
        
        # Reshape to spatial
        L = attn_map.shape[1]
        H = W = int(np.sqrt(L))
        attn_map_spatial = attn_map[0].view(H, W).cpu().numpy()
        
        # Normalize
        attn_map_spatial = (attn_map_spatial - attn_map_spatial.min()) / \
                          (attn_map_spatial.max() - attn_map_spatial.min() + 1e-8)
        
        # Visualize
        visualize_attention_weights(
            torch.from_numpy(attn_map_spatial),
            image,
            save_path,
            title="Reconstruction Attention"
        )


def visualize_patch_importance(model, image_tensor, image, save_path, patch_size=16):
    """Visualize importance của từng patch"""
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
        
        # Forward pass
        output = model(image_tensor, return_all_tokens=True)
        
        # Compute gradient-based importance (Grad-CAM style)
        tokens.requires_grad = True
        output_cls = output[:, 0]  # CLS token
        output_cls.sum().backward()
        
        # Get gradients
        gradients = tokens.grad
        # Average over channels
        importance = gradients.abs().mean(dim=-1)  # (B, N)
        
        # Remove cls token
        if importance.shape[1] > 1:
            patch_importance = importance[:, 1:]
        else:
            patch_importance = importance
        
        # Reshape to spatial
        L = patch_importance.shape[1]
        H = W = int(np.sqrt(L))
        patch_importance_spatial = patch_importance[0].view(H, W).cpu()
        
        # Visualize
        visualize_attention_weights(
            patch_importance_spatial,
            image,
            save_path,
            title="Patch Importance (Gradient-based)"
        )


def window_partition(x, window_size):
    """Window partition utility"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def main():
    parser = argparse.ArgumentParser('Visualize Attention Maps')
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
    parser.add_argument('--output_dir', type=str, default='./attention_visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--use_mab', type=utils.bool_flag, default=False,
                        help='Use MAB')
    parser.add_argument('--use_ocab', type=utils.bool_flag, default=False,
                        help='Use OCAB')
    parser.add_argument('--use_drff', type=utils.bool_flag, default=False,
                        help='Use DRFF')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    print(f"Loading image from {args.image_path}")
    image, image_tensor = load_image(args.image_path, args.image_size)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    if args.arch in models.__dict__.keys():
        use_multi_attention = args.use_mab or args.use_ocab or args.use_drff
        if use_multi_attention:
            model = models.__dict__[args.arch](
                patch_size=args.patch_size,
                return_all_tokens=True,
                use_mab=args.use_mab,
                use_ocab=args.use_ocab,
                use_drff=args.use_drff
            )
        else:
            model = models.__dict__[args.arch](
                patch_size=args.patch_size,
                return_all_tokens=True
            )
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(utils.match_statedict(state_dict), strict=False)
    model = model.cuda()
    model.eval()
    
    print("Extracting attention maps...")
    
    # Extract attention maps
    attention_maps = extract_attention_from_model(
        model, image_tensor, args.patch_size, args.image_size
    )
    
    # Visualize each attention map
    for attn_name, attn_map in attention_maps.items():
        save_path = output_dir / f"{attn_name}.png"
        visualize_attention_weights(attn_map, image, str(save_path), title=attn_name)
    
    # Gradient-based visualization
    print("Computing gradient-based importance...")
    save_path = output_dir / "gradient_importance.png"
    visualize_patch_importance(model, image_tensor, image, str(save_path), args.patch_size)
    
    print(f"\nAll visualizations saved to {output_dir}")
    print("\nCác loại attention maps:")
    print("1. wmsa_attention: Window-based Multi-Head Self-Attention")
    print("2. channel_attention: Channel Attention từ CAB")
    print("3. gradient_importance: Patch importance dựa trên gradients")
    print("\nMàu đỏ = high attention, màu xanh = low attention")


if __name__ == '__main__':
    main()

