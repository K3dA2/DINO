import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

class CelebADataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, subset_ratio=1.0):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            subset_ratio (float): Fraction of the dataset to use (0 < subset_ratio <= 1.0).
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Restrict the dataset to the specified subset
        if subset_ratio < 1.0:
            subset_size = int(len(self.data) * subset_ratio)
            self.data = self.data.sample(n=subset_size, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the image file name and attributes
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        attributes = self.data.iloc[idx, 1:].values.astype(int)

        # Convert -1 to 0 for binary tensor
        attributes = torch.tensor((attributes == 1).astype(int), dtype=torch.float32)

        # Open and preprocess the image
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, attributes
    
class MultiCropDataset(Dataset):
    def __init__(self, base_dataset, global_transform, local_transform, num_local_crops=2, final_size = 128):
        """
        Args:
            base_dataset (Dataset): The base dataset to load images and labels from.
            global_transform (callable): Transform to apply for global views.
            local_transform (callable): Transform to apply for local views.
            num_local_crops (int): Number of local crops to generate per image.
        """
        self.base_dataset = base_dataset
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.num_local_crops = num_local_crops
        self.resize = transforms.Resize((final_size, final_size))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get the original image and its labels
        image, labels = self.base_dataset[idx]

        # Apply global transformations (2 crops)
        global_crops = [self.resize(self.global_transform(image)) for _ in range(2)]

        # Apply local transformations (N crops)
        local_crops = [self.resize(self.local_transform(image)) for _ in range(self.num_local_crops)]

        # Combine the crops into one tensor
        combined_crops = global_crops + local_crops

        return combined_crops, labels
    

def visualize_augmentations(dataset, num_samples=1):
    """Visualize global and local views for a few samples."""
    for i in range(num_samples):
        global_views, local_views, _ = dataset[i]  # Extract global and local views
        
        # Visualize global views
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for j, view in enumerate(global_views):
            axes[j].imshow(view.permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Unnormalize for display
            axes[j].set_title(f"Global View {j + 1}")
            axes[j].axis("off")
        
        plt.show()

        # Visualize local views
        fig, axes = plt.subplots(1, len(local_views), figsize=(15, 5))
        for j, view in enumerate(local_views):
            axes[j].imshow(view.permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Unnormalize for display
            axes[j].set_title(f"Local View {j + 1}")
            axes[j].axis("off")
        
        plt.show()

# Visualization Function

def visualize_attention(attn_weights, num_patches, original_image):
    """
    Visualize attention maps for the [CLS] token alongside the original image.
    Args:
        attn_weights: Attention weights from the last layer.
        num_patches: Number of patches in one dimension (e.g., 16 for 16x16).
        original_image: The input image tensor of shape (3, H, W).
    """
    if attn_weights.dim() == 3:  # Case: (num_heads, seq_len, seq_len)
        cls_attn = attn_weights[:, 0, 1:]  # Exclude the [CLS] token itself
        cls_attn = cls_attn.mean(dim=0)  # Average over heads
    elif attn_weights.dim() == 4:  # Case: (batch_size, num_heads, seq_len, seq_len)
        cls_attn = attn_weights[0, :, 0, 1:]  # Exclude the [CLS] token itself
        cls_attn = cls_attn.mean(dim=0)  # Average over heads
    else:
        raise ValueError(f"Unexpected attention weights shape: {attn_weights.shape}")

    # Reshape the attention map to match the spatial layout of patches
    cls_attn = cls_attn.reshape(num_patches, num_patches).detach().cpu().numpy()
    cls_attn = (cls_attn - np.min(cls_attn)) / (np.max(cls_attn) - np.min(cls_attn))  # Normalize to [0, 1]

    # Convert the original image to a format suitable for plotting
    original_image = original_image.permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)
    original_image = (original_image - np.min(original_image)) / (np.max(original_image) - np.min(original_image))  # Normalize

    # Plot the original image and the attention map side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image
    axes[0].imshow(original_image)
    axes[0].axis('off')
    axes[0].set_title("Original Image")

    # Plot the attention map
    im = axes[1].imshow(cls_attn, cmap='viridis')
    plt.colorbar(im, ax=axes[1])  # Attach colorbar to the second plot
    axes[1].axis('off')
    axes[1].set_title("Attention Map for [CLS] Token")

    plt.tight_layout()
    plt.show()

def freeze_last_layer_grad(model, epochs_to_freeze):
    """
    Sets the gradient of the last layer of a model to None for a specified number of epochs.
    
    Args:
        model (nn.Module): The PyTorch model whose last layer's gradient should be frozen.
        epochs_to_freeze (int): Number of epochs to freeze the last layer's gradient.
    
    Returns:
        A closure that can be called within the training loop.
    """
    def apply(epoch):
        """
        Apply the gradient freeze logic based on the current epoch.
        
        Args:
            epoch (int): The current epoch number.
        """
        if epoch < epochs_to_freeze:
            # Identify the last layer
            last_layer = list(model.parameters())[-1]
            # Zero out the gradient
            if last_layer.grad is not None:
                last_layer.grad = None

    return apply


# Define the linear warmup and cosine decay scheduler
class WarmupCosineScheduler:
    def __init__(self, optimizer, base_lr, batch_size, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.base_lr = base_lr * (batch_size / 256)  # Linear scaling rule
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def get_lr(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.base_lr * 0.5 * (1 + torch.cos(torch.pi * progress))

    def step(self, epoch):
        lr = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# Define the weight decay scheduler
class WeightDecayScheduler:
    def __init__(self, optimizer, start_wd, end_wd, total_epochs):
        self.optimizer = optimizer
        self.start_wd = start_wd
        self.end_wd = end_wd
        self.total_epochs = total_epochs

    def get_wd(self, epoch):
        progress = epoch / self.total_epochs
        return self.start_wd + progress * (self.end_wd - self.start_wd)

    def step(self, epoch):
        wd = self.get_wd(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = wd

class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the root folder containing class subfolders.
            transform (callable, optional): Optional transform to be applied to images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # Get sorted class names
        self.image_paths = []
        self.labels = []

        # Scan through directories
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure valid images
                        self.image_paths.append(img_path)
                        self.labels.append(class_name)  # Store class name (not index)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")  # Convert to RGB format

        if self.transform:
            image = self.transform(image)

        return image, label  # Returning class name instead of index