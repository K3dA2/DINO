import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm 
from utils import  MultiCropDataset, AnimalDataset
from model import DINO, DINOLoss
import torch.optim as optim
import os
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_dino(epochs, train_loader, student, teacher, optimizer, criterion, momentum=0.9, start=0):
    """
    Training loop for DINO with a progress bar.
    Args:
        epochs: Number of training epochs.
        train_loader: DataLoader for training data.
        student: Student network.
        teacher: Teacher network (momentum-based).
        optimizer: Optimizer for the student network.
        criterion: DINOLoss instance.
        momentum: EMA momentum for updating teacher parameters.
    """
    teacher.eval()  # Teacher network is frozen during training
    for epoch in range(start, epochs):
        epoch_loss = 0  # Track loss for the epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")  # Initialize progress bar

        for views, _ in progress_bar:
            # Move all views to device
            views = [view.to(device) for view in views]

            # Forward pass through the networks
            teacher_outputs = teacher(views[0:2])
            student_outputs = student(views)

            # Compute DINO loss
            loss = criterion(student_outputs, teacher_outputs)
            epoch_loss += loss.item()

            # freeze last layer
            #freeze_grad(epoch)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # Update teacher with EMA and gradient clipping
            with torch.no_grad():
                for param_student, param_teacher in zip(student.parameters(), teacher.parameters()):
                    param_teacher.data = momentum * param_teacher.data + (1 - momentum) * param_student.data
            
            # Update progress bar with current loss
            progress_bar.set_postfix(loss=loss.item())

            # Logging training loss per steps
            with open("DINO-loss-per-steps.txt", "a") as file:
                file.write(f"{loss.item()}\n")

        # Save model checkpoint periodically
        if epoch % 1 == 0:
            model_filename = f'DINO-{epoch}.pth'
            model_path = os.path.join('weights/', model_filename)
            torch.save({
                'epoch': epoch + 1,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
        
        # Logging training loss
        with open("DINO-loss.txt", "a") as file:
            file.write(f"{epoch_loss / len(train_loader):.4f}\n")

        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {epoch_loss / len(train_loader):.4f}")




if __name__ == "__main__":
    # Device Setup
    device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Hyperparameters
    epochs = 100
    batch_size = 4
    learning_rate = 5e-4 * batch_size/128
    out_dim = 256  # Output dimension of projection head
    teacher_temp = 0.04
    student_temp = 0.1
    center_momentum = 0.9
    warmup_epochs = 10

    # Define augmentations for global views (high resolution)
    global_transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.4, 1.0)),  # Resizing to a large crop
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.1),  # Blur
        transforms.RandomSolarize(128, 0.2),  # Solarization
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    # Define augmentations for local views (smaller resolution)
    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.05, 0.4)),  # Smaller crops
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.1),  # Blur
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])


    # Initialize the dataset
    root_folder = ""
    base_dataset = AnimalDataset(root_folder)
    multi_crop_dataset = MultiCropDataset(base_dataset, global_transform, local_transform, num_local_crops=3)
    train_loader = DataLoader(multi_crop_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model setup
    student = DINO().to(device)
    teacher = DINO().to(device)

    # Optimizer and Loss
    optimizer = optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=0.4)
    criterion = DINOLoss(out_dim, teacher_temp, student_temp, center_momentum)

    epoch = 0
    
    # Uncomment if avialable
    '''
    # Load the checkpoint and model
    checkpoint_path = "weights/DINO-9.pth"  # Update this path if necessary
    checkpoint = torch.load(checkpoint_path)
    student.load_state_dict(checkpoint['student_state_dict'])
    teacher.load_state_dict(checkpoint['teacher_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get("epoch",0)
    '''

    for param in teacher.parameters():
        param.requires_grad = False

    # Print param count
    print("Total parameters: ", count_parameters(student))

    # Freeze last layer gradient for the first N epochs
    #freeze_grad = freeze_last_layer_grad(student, epochs_to_freeze=1)

    # Train the model
    train_dino(epochs, train_loader, student, teacher, optimizer, criterion, start=epoch)
