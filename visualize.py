import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import colorsys
import random
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from model import DINO  # Make sure this model includes get_last_selfattention method

# Denormalization function for visualization
def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    # Create a copy of the tensor to avoid modifying the original
    tensor = tensor.clone().detach()
    # tensor shape: (C,H,W)
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Functions from DINO visualization script
def apply_mask(image, mask, color, alpha=0.5):
    # Create a copy to avoid modifying the original
    result = image.copy()
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return result

def random_colors(N, bright=True):
    """Generate random colors."""
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, mask, fname=None, figsize=(5, 5), blur=False, contour=True, alpha=0.5, save=False):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    
    if save and fname:
        fig.savefig(fname)
        plt.close(fig)
        print(f"{fname} saved.")
        return None
    else:
        plt.tight_layout()
        return fig

# Define your dataset
class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # Get sorted class names
        self.image_paths = []
        self.labels = []
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(img_path)
                        self.labels.append(class_name)
                        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path  # Also return the path for saving visualizations

def visualize_attention(save_images=True, num_images=1):
    # Set up transforms and dataloader
    root_folder = "/Users/ayanfe/Documents/Datasets/mammals"  # Update this path as needed
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # DINO typically uses larger images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = AnimalDataset(root_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)  # Set batch size to 1 for better control

    # Load the checkpoint and model
    checkpoint_path = "weights/DINO-19.pth"  # Update this path if necessary
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create a DINO model instance
    model = DINO()  # Using default parameters; adjust if needed
    model.load_state_dict(checkpoint['student_state_dict'])
    model.eval()

    # Determine patch size
    patch_size = model.backbone.patch_embed.kernel_size[0]
    output_dir = "attention_visualizations"
    if save_images:
        os.makedirs(output_dir, exist_ok=True)

    # Loop through some images in the dataset
    for i, (images, labels, paths) in enumerate(dataloader):
        if i >= num_images:  # Process specified number of images
            break
        
        img_size = images.shape[-1]  # assuming square images
        # Calculate feature map size
        w_featmap = img_size // patch_size
        h_featmap = img_size // patch_size
        
        with torch.no_grad():
            # Get full self-attention from the model
            attentions = model.get_last_selfattention(images)
        
        # Print attention statistics for debugging
        print(f"Attention shape: {attentions.shape}")
        print(f"Attention stats - Min: {attentions.min().item():.6f}, Max: {attentions.max().item():.6f}")
        print(f"Attention stats - Mean: {attentions.mean().item():.6f}, Std: {attentions.std().item():.6f}")
        
        # Number of attention heads
        nh = attentions.shape[1]
        
        # 1. Extract CLS token attention to patches (excluding CLS token's attention to itself)
        # Shape: [num_heads, num_patches]
        cls_attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        
        # 2. Create proper heatmaps (reshape to spatial dimensions)
        cls_attention_maps = cls_attentions.reshape(nh, w_featmap, h_featmap)
        
        # 3. Also get the full attention maps (all tokens attending to all other tokens)
        # This can be used for a different visualization
        full_attention_maps = attentions[0, :, :, :]  # [num_heads, N, N]
        
        # 4. Interpolate to image size for visualization
        cls_attention_maps_upsampled = torch.nn.functional.interpolate(
            cls_attention_maps.unsqueeze(0), 
            scale_factor=patch_size, 
            mode="bilinear"
        )[0].cpu().numpy()
        
        # Image metadata
        img_name = os.path.basename(paths[0])
        save_prefix = os.path.join(output_dir, f"{i}_{labels[0]}_{os.path.splitext(img_name)[0]}")
        
        # Convert tensor to PIL for saving/display
        img_pil = transforms.ToPILImage()(denormalize(images[0].clone()).clamp(0, 1))
        
        # Save original image if requested
        if save_images:
            img_pil.save(f"{save_prefix}_original.png")
            print(f"Saved original image to: {save_prefix}_original.png")
        
        # Convert to numpy for visualization
        image_np = np.array(img_pil)
        
        # Display original image if not saving
        if not save_images:
            plt.figure(figsize=(5, 5))
            plt.imshow(image_np)
            plt.title(f"Original: {labels[0]}")
            plt.axis('off')
            plt.show()
        
        # Process and visualize attention maps for each head
        for j in range(nh):
            # Normalize the attention map to [0, 1] range for better visualization
            attn_map = cls_attention_maps_upsampled[j]
            attn_min, attn_max = attn_map.min(), attn_map.max()
            if attn_max > attn_min:  # Avoid division by zero
                attn_map_norm = (attn_map - attn_min) / (attn_max - attn_min)
            else:
                attn_map_norm = attn_map
            
            # Raw attention heatmap visualization
            if save_images:
                plt.figure(figsize=(6, 5))
                plt.imshow(attn_map_norm, cmap='viridis')
                plt.colorbar(label='Normalized Attention')
                plt.title(f"Head {j} - Raw Attention")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f"{save_prefix}_attn_raw_head{j}.png")
                plt.close()
                print(f"Saved raw attention to: {save_prefix}_attn_raw_head{j}.png")
            else:
                plt.figure(figsize=(6, 5))
                plt.imshow(attn_map_norm, cmap='viridis')
                plt.colorbar(label='Normalized Attention')
                plt.title(f"Head {j} - Raw Attention")
                plt.axis('off')
                plt.show()
            
            # Overlay visualization (attention heatmap over the original image)
            if save_images:
                plt.figure(figsize=(6, 5))
                plt.imshow(image_np)
                plt.imshow(attn_map_norm, alpha=0.7, cmap='jet')
                plt.colorbar(label='Attention Strength')
                plt.title(f"Head {j} - Attention Overlay")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f"{save_prefix}_attn_overlay_head{j}.png")
                plt.close()
                print(f"Saved overlay to: {save_prefix}_attn_overlay_head{j}.png")
            else:
                plt.figure(figsize=(6, 5))
                plt.imshow(image_np)
                plt.imshow(attn_map_norm, alpha=0.7, cmap='jet')
                plt.colorbar(label='Attention Strength')
                plt.title(f"Head {j} - Attention Overlay")
                plt.axis('off')
                plt.show()
            
            # Also visualize thresholded attention for comparison (DINO style)
            # Apply threshold at 50% of attention mass
            threshold = 0.5
            
            # Sort attention values
            flat_attn = cls_attentions[j].clone()
            sorted_attn, sorted_indices = torch.sort(flat_attn)
            cumulative_attn = torch.cumsum(sorted_attn, dim=0)
            cumulative_attn = cumulative_attn / cumulative_attn[-1].item()  # Normalize to sum to 1
            
            # Find threshold index
            threshold_idx = torch.searchsorted(cumulative_attn, 1.0 - threshold)
            threshold_value = sorted_attn[threshold_idx]
            
            # Create binary mask
            binary_mask = (flat_attn > threshold_value).float()
            binary_mask = binary_mask.reshape(w_featmap, h_featmap)
            
            # Upsample to image size
            binary_mask_upsampled = torch.nn.functional.interpolate(
                binary_mask.unsqueeze(0).unsqueeze(0), 
                scale_factor=patch_size, 
                mode="nearest"
            )[0, 0].cpu().numpy()
            
            # Visualize thresholded mask with original DINO visualization
            if save_images:
                display_instances(
                    image_np,
                    binary_mask_upsampled, 
                    fname=f"{save_prefix}_mask_th{threshold}_head{j}.png",
                    figsize=(5, 5),
                    blur=False,
                    save=True
                )
                print(f"Saved thresholded mask to: {save_prefix}_mask_th{threshold}_head{j}.png")
            else:
                fig = display_instances(
                    image_np,
                    binary_mask_upsampled, 
                    figsize=(5, 5),
                    blur=False,
                    save=False
                )
                plt.figure(fig.number)
                plt.title(f"Head {j} - Thresholded Attention Mask")
                plt.show()
        
        print(f"Processed image {i+1}: {labels[0]} - {img_name}")

# Call the function - by default it displays images instead of saving them
if __name__ == "__main__":
    # To display images (default):
    visualize_attention()
    
    # To save images instead:
    # visualize_attention(save_images=True, num_images=5)