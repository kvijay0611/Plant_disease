#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pickle
import warnings 

warnings.filterwarnings('ignore')


# In[2]:


import matplotlib.pyplot as plt
import random
import os
from PIL import Image
import numpy as np

def display_disease_samples(data_dir, plants=None, num_cols=5):
    disease_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

    if plants is not None:
        disease_folders = [f for f in disease_folders if any(p in f for p in plants)]

    num_diseases = len(disease_folders)
    num_rows = (num_diseases + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
    axes = axes.flatten() if num_rows > 1 else axes

    for i, disease_folder in enumerate(disease_folders):
        folder_path = os.path.join(data_dir, disease_folder)

        img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if img_files:
            img_path = os.path.join(folder_path, random.choice(img_files))
            img = Image.open(img_path).convert('RGB')

            disease_name = disease_folder.replace('_', ' ')

            axes[i].imshow(img)
            axes[i].set_title(disease_name, fontsize=12)
            axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

print("🌿 Sample images from different plant disease categories:")
display_disease_samples("/kaggle/input/plantdisease/PlantVillage")


# ## Custom Dataset Class for Plant Disease Images
# 
# The `PlantDiseaseDataset` class is a custom dataset handler designed to load and preprocess plant disease images. It extends PyTorch's `Dataset` class, providing a structured way to manage image data and their corresponding labels.
# 
# ### Key Features:
# 
# - **Initialization**: Accepts image paths, labels, and optional transformations. This allows for flexible data handling and preprocessing.
# - **Length Method**: Returns the total number of samples, facilitating iteration over the dataset.
# - **Get Item Method**: Retrieves an image and its label at a specified index, applying any transformations to prepare the data for model input.
# 
# This class is crucial for efficiently managing the data pipeline in our plant disease classification task.

# In[3]:


# Dataset Class
class PlantDiseaseDataset(Dataset):
    """Custom Dataset for loading plant disease images"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image and apply transformations
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


# ## Convolutional Neural Network for Plant Disease Classification
# 
# In this section, we introduce our custom Convolutional Neural Network (CNN) architecture, `PlantDiseaseModel`, designed specifically for classifying plant diseases. This model leverages multiple convolutional layers to extract features from images, followed by fully connected layers to perform classification.
# 
# ### Architecture Overview
# 
# The `PlantDiseaseModel` is composed of several key components:
# 
# - **Convolutional Blocks**: The model consists of five convolutional blocks, each containing a convolutional layer, batch normalization, ReLU activation, and max pooling. These blocks progressively extract and refine features from the input images.
# 
# - **Global Average Pooling**: After the convolutional layers, a global average pooling layer reduces the spatial dimensions, summarizing the feature maps into a single vector per feature map.
# 
# - **Fully Connected Layers**: The final layers of the model are fully connected, transforming the pooled features into class probabilities. A dropout layer is included to prevent overfitting.
# 
# ### Why This Architecture?
# 
# This architecture is designed to balance complexity and performance, making it suitable for the task of plant disease classification. The use of multiple convolutional layers allows the model to learn intricate patterns in the data, while the dropout layer helps mitigate overfitting, ensuring robust performance on unseen data.

# In[4]:


from torchinfo import summary

class PlantDiseaseModel(nn.Module):
    """Convolutional Neural Network for plant disease classification"""
    def __init__(self, num_classes, dropout_rate=0.5):
        super(PlantDiseaseModel, self).__init__()
        # Convolutional Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Convolutional Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Convolutional Block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Convolutional Block 4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Convolutional Block 5
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully Connected Layers
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.global_avg_pool(x)
        x = self.fc_block(x)
        return x

print(summary(PlantDiseaseModel(15), input_size=(1, 3, 224, 224)))


# ## Early Stopping Utility
# 
# To enhance our model training process, we implement an `EarlyStopping` utility. This mechanism helps prevent overfitting by monitoring the validation loss and halting training when the loss stops improving.
# 
# ### How Early Stopping Works
# 
# - **Patience**: The `patience` parameter defines how many epochs to wait for an improvement in validation loss before stopping the training. If the loss does not improve for a specified number of epochs, training is terminated.
# 
# - **Minimum Delta**: The `min_delta` parameter sets the minimum change in the monitored quantity to qualify as an improvement. This helps in ignoring minor fluctuations in validation loss.
# 
# - **Model Checkpointing**: When a new best validation loss is observed, the model's state is saved to a specified path (`save_path`). This ensures that the best-performing model is retained.

# In[5]:


# Early Stopping Utility
class EarlyStopping:
    """Early stopping handler to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.001, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model
            torch.save(model.state_dict(), self.save_path)
            print(f"[INFO] Model checkpoint saved to {self.save_path}")
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("[INFO] Early stopping triggered.")
                return True
        return False


# ## Data Loading and Preparation
# 
# This section includes functions to load images from a directory structure and prepare them for training. It also handles label encoding and data splitting into training, validation, and test sets.

# In[6]:


# Data Loading and Preparation Functions
def load_images(directory_root):
    """Load images and their labels from directory structure"""
    image_list, label_list = [], []
    print("[INFO] Loading images...")

    for disease_folder in os.listdir(directory_root):
        disease_folder_path = os.path.join(directory_root, disease_folder)
        if not os.path.isdir(disease_folder_path):
            continue

        for img_name in os.listdir(disease_folder_path):
            if img_name.startswith("."):
                continue
            img_path = os.path.join(disease_folder_path, img_name)
            if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_list.append(img_path)
                label_list.append(disease_folder)

    print("[INFO] Image loading completed")
    print(f"Total images: {len(image_list)}")
    return image_list, label_list

def prepare_data(directory_root, image_size=(256, 256), batch_size=32, test_size=0.3, valid_ratio=0.5, random_state=42):
    """Prepare data loaders and label encoder"""
    # Load images and labels
    image_paths, labels = load_images(directory_root)

    # Encode labels as integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Save label encoder for inference
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Save class names for reference
    class_names = list(label_encoder.classes_)
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)

    # Train, validation, and test splits
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels_encoded, test_size=test_size, random_state=random_state, stratify=labels_encoded
    )
    valid_paths, test_paths, valid_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=valid_ratio, random_state=random_state, stratify=temp_labels
    )

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(valid_paths)}")
    print(f"Test samples: {len(test_paths)}")

    # Data Transformations
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Save image transformation for inference
    with open('inference_transform.pkl', 'wb') as f:
        pickle.dump(valid_test_transform, f)

    # Create datasets with appropriate transformations
    train_dataset = PlantDiseaseDataset(train_paths, train_labels, transform=train_transform)
    valid_dataset = PlantDiseaseDataset(valid_paths, valid_labels, transform=valid_test_transform)
    test_dataset = PlantDiseaseDataset(test_paths, test_labels, transform=valid_test_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, len(class_names)


# In[7]:


def plot_dataset_distribution(data_dir):
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

    counts = {}
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        image_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        counts[folder] = image_count

    plant_data = {}
    for folder, count in counts.items():
        # Extract plant type (e.g., "Tomato" from "Tomato_Bacterial_spot")
        if "__" in folder:
            parts = folder.split("__")
            plant = parts[0].replace("_", " ")
        else:
            plant = folder.split("_")[0]

        # Check if healthy or diseased
        if "healthy" in folder.lower():
            status = "Healthy"
        else:
            status = "Diseased"

        # Organize data by plant type and status
        if plant not in plant_data:
            plant_data[plant] = {"Healthy": 0, "Diseased": 0}
        plant_data[plant][status] += count

    # Create plot with enhanced styling
    plt.style.use('seaborn-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    # Color palette
    healthy_color = '#4CAF50'  # Modern green
    diseased_color = '#FF5722'  # Vibrant orange
    pie_colors = plt.cm.tab20.colors  # More distinct colors

    # Plot 1: Enhanced stacked bar chart
    plants = list(plant_data.keys())
    healthy_counts = [plant_data[p]["Healthy"] for p in plants]
    diseased_counts = [plant_data[p]["Diseased"] for p in plants]

    # Create gradient effect for bars
    bar1 = ax1.bar(plants, healthy_counts, label='Healthy', 
                   color=healthy_color, edgecolor='#2E7D32', linewidth=1.5)
    bar2 = ax1.bar(plants, diseased_counts, bottom=healthy_counts, 
                   label='Diseased', color=diseased_color, edgecolor='#BF360C', linewidth=1.5)

    # Enhanced annotations
    for i, plant in enumerate(plants):
        total = healthy_counts[i] + diseased_counts[i]
        ax1.text(i, total + 50, f'{total}', ha='center', 
                fontsize=10, fontweight='bold', color='#37474F')
        # Add percentage labels inside bars
        healthy_pct = healthy_counts[i]/total * 100
        ax1.text(i, healthy_counts[i]/2, f'{healthy_pct:.1f}%', 
                ha='center', va='center', color='white', fontsize=9)
        diseased_pct = diseased_counts[i]/total * 100
        ax1.text(i, healthy_counts[i] + diseased_counts[i]/2, f'{diseased_pct:.1f}%', 
                ha='center', va='center', color='white', fontsize=9)

    ax1.set_title('Healthy vs Diseased Distribution by Plant Type\n', 
                 fontsize=16, fontweight='bold', color='#2E4053')
    ax1.set_xlabel('Plant Type', fontsize=13, labelpad=15)
    ax1.set_ylabel('Number of Images', fontsize=13, labelpad=15)
    ax1.tick_params(axis='x', rotation=45, labelsize=11)
    ax1.legend(frameon=True, shadow=True, fontsize=12)

    # Add subtle grid
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4)

    # Plot 2: Enhanced pie chart
    plant_totals = {p: plant_data[p]["Healthy"] + plant_data[p]["Diseased"] for p in plants}
    wedges, texts, autotexts = ax2.pie(plant_totals.values(), labels=plant_totals.keys(), 
                                      autopct='%1.1f%%', startangle=90,
                                      colors=pie_colors, 
                                      wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
                                      textprops={'fontsize': 11})

    # Improve percentage formatting
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # Add donut effect
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax2.add_artist(centre_circle)

    ax2.set_title('Image Distribution by Plant Type\n', 
                 fontsize=16, fontweight='bold', color='#2E4053')

    plt.tight_layout(pad=3.0)

    # Summary statistics with enhanced formatting
    total_images = sum(plant_totals.values())
    total_healthy = sum(healthy_counts)
    total_diseased = sum(diseased_counts)

    print(f"\n{'📊'*3} Dataset Summary {'📊'*3}")
    print(f"\n🔸 Total images: \033[1m{total_images:,}\033[0m")
    print(f"🔸 Healthy samples: \033[1m{total_healthy:,}\033[0m ({total_healthy/total_images:.1%})")
    print(f"🔸 Diseased samples: \033[1m{total_diseased:,}\033[0m ({total_diseased/total_images:.1%})")
    print(f"🔸 Number of plant types: \033[1m{len(plants)}\033[0m")
    print(f"🔸 Number of disease categories: \033[1m{len(folders) - len(healthy_counts)}\033[0m\n")

plot_dataset_distribution("/kaggle/input/plantdisease/PlantVillage")


# In[8]:


def show_augmentations(data_dir, num_plants=3):
    disease_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    selected_folders = random.sample(disease_folders, min(num_plants, len(disease_folders)))

    # Define augmentations to display like the training used one.
    augmentations = [
        ("Original", transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])),
        ("Horizontal Flip", transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor()
        ])),
        ("Rotation (30°)", transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(30),
            transforms.ToTensor()
        ])),
        ("Color Jitter", transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])),
        ("Combined", transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor()
        ]))
    ]

    fig, axes = plt.subplots(len(selected_folders), len(augmentations), figsize=(18, 4 * len(selected_folders)))

    for i, folder in enumerate(selected_folders):
        folder_path = os.path.join(data_dir, folder)

        img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not img_files:
            continue

        img_path = os.path.join(folder_path, random.choice(img_files))
        original_img = Image.open(img_path).convert('RGB')

        for j, (aug_name, transform) in enumerate(augmentations):
            img_tensor = transform(original_img)

            img_np = img_tensor.permute(1, 2, 0).numpy()

            ax = axes[i, j] if len(selected_folders) > 1 else axes[j]
            ax.imshow(img_np)

            if i == 0:
                ax.set_title(aug_name, fontsize=12)

            if j == 0:
                disease_name = folder.replace('_', ' ')
                ax.set_ylabel(disease_name, fontsize=10)

            ax.axis('off')

    plt.tight_layout()
    plt.suptitle("Data Augmentation Techniques for Plant Disease Images", fontsize=20, y=1.0)
    plt.show()

show_augmentations("/kaggle/input/plantdisease/PlantVillage")


# ## Training and Evaluation Functions
# 
# These functions handle the training and evaluation of the model. They include utilities for calculating loss and accuracy, as well as saving learning curves for analysis.

# In[9]:


# Training and Evaluation Functions
def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model on validation or test set"""
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(enumerate(data_loader), desc="Evaluating", total=len(data_loader))
        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({"Val Loss": loss.item(), "Accuracy": correct / total * 100})

    val_loss /= len(data_loader)
    accuracy = correct / total * 100
    return val_loss, accuracy, np.array(all_preds), np.array(all_labels)

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler=None, 
                epochs=10, early_stopping=None, device="cpu"):
    """Train the model with optional early stopping and learning rate scheduler"""
    model.to(device)
    train_losses, valid_losses, valid_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{epochs}", 
                           total=len(train_loader))

        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Train Loss": loss.item()})

        # Record training loss
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        val_loss, val_accuracy, _, _ = evaluate_model(model, valid_loader, criterion, device)
        valid_losses.append(val_loss)
        valid_accuracies.append(val_accuracy)

        # Print epoch summary
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
              f"Val Accuracy = {val_accuracy:.2f}%")

        # Learning rate scheduler step
        if scheduler:
            scheduler.step(val_loss)

        # Early stopping
        if early_stopping and early_stopping(val_loss, model):
            print("[INFO] Early stopping triggered.")
            break

    # Save the learning curves
    save_learning_curves(train_losses, valid_losses, valid_accuracies)

    return train_losses, valid_losses, valid_accuracies

def save_learning_curves(train_losses, valid_losses, valid_accuracies):
    """Save learning curves as a plot"""
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()


# ## Prediction and Main Training Function
# 
# The prediction function allows for inference on single images, while the main training function orchestrates the entire training process, including data preparation, model training, and evaluation.

# In[10]:


# Prediction function for inference
def predict_image(model, image_path, transform, device, label_encoder=None):
    """Make prediction on a single image"""
    model.eval()

    # Open and transform the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    predicted_idx = predicted.item()
    confidence = probabilities[0][predicted_idx].item() * 100

    if label_encoder:
        predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
        return predicted_class, confidence, probabilities[0].cpu().numpy()
    else:
        return predicted_idx, confidence, probabilities[0].cpu().numpy()

# Main function to train the model
def train(data_dir, model_save_path="best_model.pth", batch_size=32, 
          epochs=30, learning_rate=0.001, image_size=(256, 256)):
    """Main function to train and save the model and necessary files for deployment"""
    # Prepare data
    train_loader, valid_loader, test_loader, num_classes = prepare_data(
        data_dir, image_size=image_size, batch_size=batch_size
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, loss function, optimizer and scheduler
    model = PlantDiseaseModel(num_classes=num_classes, dropout_rate=0.5)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    early_stopping = EarlyStopping(patience=7, min_delta=0.001, save_path=model_save_path)

    # Print model summary
    print(f"Model created with {num_classes} output classes")

    # Train the model
    train_model(
        model=model, 
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        criterion=criterion, 
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        early_stopping=early_stopping,
        device=device
    )

    # Load the best model
    model.load_state_dict(torch.load(model_save_path))

    # Evaluate on test set
    print("\n[INFO] Evaluating the model on the test set...")
    test_loss, test_accuracy, predictions, true_labels = evaluate_model(
        model, test_loader, criterion, device
    )
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Save model architecture for inference
    dummy_input = torch.randn(1, 3, *image_size).to(device)
    torch.onnx.export(model, dummy_input, "plant_disease_model.onnx")

    # Save model config
    model_config = {
        "image_size": image_size,
        "num_classes": num_classes,
        "model_path": model_save_path,
        "label_encoder_path": "label_encoder.pkl",
        "transform_path": "inference_transform.pkl",
        "class_names_path": "class_names.json"
    }

    with open("model_config.json", "w") as f:
        json.dump(model_config, f)

    print("[INFO] Training completed and all necessary files saved for deployment.")
    return model, model_config



# ## Main Execution Block
# 
# In this section, we define the main execution block of our script. This block initializes the necessary parameters and calls the `train` function to start the training process for our plant disease classification model. 
# 
# The parameters include:
# - `data_dir`: The directory containing the dataset.
# - `model_path`: The path where the best model will be saved.
# - `batch_size`: The number of samples processed before the model is updated.
# - `epochs`: The number of complete passes through the training dataset.
# - `learning_rate`: The step size at each iteration while moving toward a minimum of the loss function.
# 
# This block ensures that the training process is initiated when the script is run directly.

# In[11]:


if __name__ == "__main__":
    data_dir = "/kaggle/input/plantdisease/PlantVillage"
    model_path = "best_model.pth"
    batch_size = 32
    epochs = 30
    learning_rate = 0.00065

    model, model_config = train(
        data_dir=data_dir,
        model_save_path=model_path,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate
    )


# In[12]:


def visualize_learning_curves():
    try:
        # Load training history if it exists
        with open('learning_curves.png', 'rb') as f:
            plt.figure(figsize=(12, 5))
            img = plt.imread('learning_curves.png')
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    except FileNotFoundError:
        print("Learning curves not found. Train the model first.")

print("📊 Visualizing Model Training Progress:")
visualize_learning_curves()


# In[13]:


def evaluate_by_plant_type(model, test_loader, label_encoder, device):
    """Evaluate model performance separately for each plant type with enhanced visualization"""
    model.eval()

    # Prepare containers for per-class metrics
    class_correct = {}
    class_total = {}

    # Get all predictions
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Store predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update per-class counts
            for i, label in enumerate(labels):
                label_idx = label.item()
                label_name = label_encoder.inverse_transform([label_idx])[0]

                if label_name not in class_correct:
                    class_correct[label_name] = 0
                    class_total[label_name] = 0

                class_total[label_name] += 1
                if preds[i] == label:
                    class_correct[label_name] += 1

    # Extract plant types from class names
    plants = {}
    for class_name in class_correct.keys():
        if "__" in class_name:
            plant = class_name.split("__")[0].replace("_", " ")
        else:
            plant = class_name.split("_")[0]

        if plant not in plants:
            plants[plant] = {"correct": 0, "total": 0}

        plants[plant]["correct"] += class_correct[class_name]
        plants[plant]["total"] += class_total[class_name]

    # Compute accuracy per plant type
    plant_accuracy = {p: (stats["correct"] / stats["total"]) * 100 
                     for p, stats in plants.items()}

    # Sort plants by accuracy for better visual comparison
    sorted_plants = dict(sorted(plant_accuracy.items(), key=lambda x: x[1], reverse=True))

    # Enhanced Visualization
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)

    # Improved color configuration
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_plants)))

    # Calculate average accuracy
    avg_accuracy = np.mean(list(plant_accuracy.values()))

    # Create bars with enhanced styling
    plants_list = list(sorted_plants.keys())
    accuracies = list(sorted_plants.values())
    totals = [plants[p]["total"] for p in plants_list]

    # Create gradient background
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#ffffff')

    # Create enhanced bars
    bars = ax.bar(plants_list, accuracies, color=colors, edgecolor='#505050', 
                 linewidth=1, alpha=0.85, width=0.7)

    # Add drop shadow effect to bars
    for bar in bars:
        x, y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        shadow = plt.Rectangle((x+0.03, y-0.03), w, h, color='#00000022', zorder=0)
        ax.add_patch(shadow)

    # Add annotations with improved styling
    for bar, acc, total in zip(bars, accuracies, totals):
        height = bar.get_height()
        # Add accuracy labels
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{acc:.1f}%', 
               ha='center', va='bottom',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="grey", alpha=0.8))

        # Add sample size labels
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
               f'n={total}', 
               ha='center', va='center',
               fontsize=10, color='#303030',
               fontweight='bold', rotation=0)

    # Add reference lines and styling
    ax.axhline(avg_accuracy, color='#e74c3c', linestyle='-', linewidth=2.5, alpha=0.7)
    ax.axhline(avg_accuracy, color='#c0392b', linestyle='-', linewidth=1, alpha=1)

    # Add average line label with enhanced styling
    ax.text(len(plants_list)-0.5, avg_accuracy + 3,
           f' Average: {avg_accuracy:.1f}%',
           color='#c0392b', fontsize=13, ha='right', va='bottom',
           fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="#c0392b", alpha=0.8))

    # Configure axes and labels with enhanced styling
    ax.set_title(f'Model Accuracy by Plant Type\n{model.__class__.__name__} Performance Analysis', 
                fontsize=18, pad=20, fontweight='bold', color='#2c3e50')

    ax.set_xlabel('Plant Type', fontsize=14, labelpad=15, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Accuracy (%)', fontsize=14, labelpad=15, fontweight='bold', color='#2c3e50')

    # Add a subtle box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')
        spine.set_linewidth(1)

    # Enhanced tick parameters
    ax.tick_params(axis='x', rotation=45, labelsize=12, pad=5, colors='#2c3e50')
    ax.tick_params(axis='y', labelsize=12, pad=5, colors='#2c3e50')
    ax.set_ylim(0, max(accuracies) * 1.15)

    # Add customized grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#95a5a6')
    ax.set_axisbelow(True)

    # Add a subtle top performance indicator
    top_performer = plants_list[0]
    top_accuracy = accuracies[0]
    ax.text(0, max(accuracies) * 1.1,
           f"Top Performer: {top_performer} ({top_accuracy:.1f}%)",
           fontsize=12, ha='left', color='#27ae60',
           bbox=dict(boxstyle="round,pad=0.3", fc='#f8f9fa', ec="#2ecc71", alpha=0.8))

    # Add watermark or model info
    fig.text(0.95, 0.02, f"{model.__class__.__name__}", 
             fontsize=10, color='gray', ha='right', va='bottom', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return plant_accuracy


# In[14]:


import torch
import pickle
import os
import random
from PIL import Image

# Load model and necessary components
model_path = "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names and create model
with open('class_names.json', 'r') as f:
    class_names = json.load(f)
num_classes = len(class_names)

model = PlantDiseaseModel(num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model.to(device)

# Load label encoder and transform
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('inference_transform.pkl', 'rb') as f:
    transform = pickle.load(f)

print(f"✅ Model loaded with {num_classes} classes")
print(f"✅ Using device: {device}")

# Cell 3: Evaluate Model Performance by Plant Type
data_dir = "/kaggle/input/plantdisease/PlantVillage"
batch_size = 32

_, _, test_loader, _ = prepare_data(
    data_dir, 
    image_size=(256, 256), 
    batch_size=batch_size,
    test_size=0.3,
    valid_ratio=0.5
)

print("\n📊 Evaluating model performance by plant type...")
plant_accuracy = evaluate_by_plant_type(model, test_loader, label_encoder, device)


# In[15]:


def apply_gradcam(model, img_path, transform, label_encoder, device, layer_name='conv_block5'):
    try:
        import cv2
    except ImportError:
        print("OpenCV (cv2) is required for Grad-CAM visualization. Please install it with: !pip install opencv-python")
        return

    model.eval()

    # Hook for the selected layer
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    # Register hooks
    if layer_name == 'conv_block5':
        target_layer = model.conv_block5[0]  # First conv layer of the last block
    elif layer_name == 'conv_block4':
        target_layer = model.conv_block4[0]
    else:
        target_layer = model.conv_block3[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    try:
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)

        # Forward pass
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        pred_class = label_encoder.inverse_transform([pred_idx])[0]

        # Backward pass for the predicted class
        model.zero_grad()
        output[:, pred_idx].backward()

        # Generate Grad-CAM
        if activations is not None and gradients is not None:
            # Pool gradients across the channels
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

            # Weight activation maps by gradients
            for i in range(activations.size(1)):
                activations[:, i, :, :] *= pooled_gradients[i]

            # Average over channels
            heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()

            # ReLU on heatmap
            heatmap = np.maximum(heatmap, 0)

            # Normalize heatmap
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)

            # Resize heatmap to original image size
            original_img = np.array(img)
            heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

            # Apply colormap to heatmap
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Superimpose heatmap on original image
            superimposed = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

            # Create figure with original and heatmap
            plt.figure(figsize=(15, 5))

            # Plot original image
            plt.subplot(1, 3, 1)
            plt.imshow(original_img)
            plt.title("Original Image", fontsize=14)
            plt.axis('off')

            # Plot heatmap
            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
            plt.title("Grad-CAM Heatmap", fontsize=14)
            plt.axis('off')

            # Plot superimposed
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
            plt.title(f"Prediction: {pred_class}", fontsize=14)
            plt.axis('off')

            plt.tight_layout()
            plt.show()
        else:
            print("Could not generate activations or gradients")
    finally:
        # Always remove hooks to prevent memory leaks
        forward_handle.remove()
        backward_handle.remove()

# Cell 5: Apply Grad-CAM to sample images
def generate_gradcam_visualizations(num_samples=2):
    print("\n🔍 Generating Grad-CAM visualizations...")
    sample_images = []

    for disease_folder in os.listdir(data_dir):
        disease_folder_path = os.path.join(data_dir, disease_folder)
        if not os.path.isdir(disease_folder_path):
            continue

        img_files = [f for f in os.listdir(disease_folder_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if img_files:
            selected_img = os.path.join(disease_folder_path, random.choice(img_files))
            sample_images.append((selected_img, disease_folder))

    # Apply GradCAM to a couple of sample images
    if sample_images:
        samples_to_visualize = random.sample(sample_images, min(num_samples, len(sample_images)))
        for i, (img_path, true_label) in enumerate(samples_to_visualize):
            print(f"\nVisualizing sample {i+1} - {true_label}...")
            apply_gradcam(model, img_path, transform, label_encoder, device)
    else:
        print("No sample images found.")

# Generate visualizations for 2 random samples
generate_gradcam_visualizations(num_samples=2)


# ## Inference

# In[16]:


def interactive_disease_diagnosis(model_path, label_encoder_path, transform_path, sample_images_dir):
    """
    Create an interactive display showing disease diagnosis and treatment recommendations

    Arguments:
    model_path -- path to the trained model
    label_encoder_path -- path to the saved label encoder
    transform_path -- path to the saved transform
    sample_images_dir -- directory containing sample images
    """
    import json
    import os
    import pickle
    import random
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from matplotlib.gridspec import GridSpec

    # Load model
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    model = PlantDiseaseModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Load label encoder and transform
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    with open(transform_path, 'rb') as f:
        transform = pickle.load(f)

    # Treatment recommendations for common plant diseases
    treatment_recommendations = {
        "Tomato_Bacterial_spot": [
            "Remove and destroy infected plants",
            "Rotate crops (avoid planting tomatoes in the same location for 2-3 years)",
            "Use copper-based fungicides",
            "Ensure proper spacing between plants for good air circulation"
        ],
        "Tomato_Early_blight": [
            "Remove infected leaves immediately",
            "Apply fungicides containing chlorothalonil or copper",
            "Mulch around the base of plants",
            "Water at soil level rather than on foliage"
        ],
        "Tomato_Late_blight": [
            "Remove and destroy infected plants",
            "Apply fungicides proactively before symptoms appear",
            "Improve air circulation around plants",
            "Avoid overhead irrigation"
        ],
        "Tomato_Leaf_Mold": [
            "Increase spacing between plants to improve air circulation",
            "Apply fungicides containing chlorothalonil or copper",
            "Remove infected leaves",
            "Keep foliage dry by watering at the base"
        ],
        "Tomato_Septoria_leaf_spot": [
            "Remove infected leaves",
            "Apply fungicides containing chlorothalonil or copper",
            "Rotate crops",
            "Mulch around plants to prevent spores splashing from soil"
        ],
        "Tomato_Spider_mites_Two_spotted_spider_mite": [
            "Spray plants with strong streams of water to dislodge mites",
            "Apply insecticidal soap or neem oil",
            "Introduce predatory mites",
            "Increase humidity around plants"
        ],
        "Tomato__Target_Spot": [
            "Remove infected plant debris",
            "Apply fungicides",
            "Improve air circulation",
            "Avoid overhead watering"
        ],
        "Tomato__Tomato_YellowLeaf__Curl_Virus": [
            "No cure available - remove and destroy infected plants",
            "Control whitefly populations (vectors)",
            "Use reflective mulches to repel whiteflies",
            "Plant resistant varieties"
        ],
        "Tomato__Tomato_mosaic_virus": [
            "No cure available - remove and destroy infected plants",
            "Wash hands and tools after handling infected plants",
            "Control aphid populations (vectors)",
            "Plant resistant varieties"
        ],
        "Potato___Early_blight": [
            "Remove infected leaves",
            "Apply fungicides containing chlorothalonil",
            "Maintain good soil fertility",
            "Ensure proper hilling to protect tubers"
        ],
        "Potato___Late_blight": [
            "Apply fungicides preventatively",
            "Remove volunteer potato plants",
            "Harvest tubers during dry weather",
            "Ensure proper storage conditions for harvested potatoes"
        ],
        "Pepper__bell___Bacterial_spot": [
            "Remove infected plant debris",
            "Rotate crops",
            "Apply copper-based sprays",
            "Use disease-free seeds"
        ]
    }

    # Default recommendation for healthy plants
    default_healthy_practices = [
        "Maintain proper watering schedule",
        "Ensure adequate sunlight",
        "Fertilize appropriately for plant type",
        "Monitor regularly for signs of disease"
    ]

    # Helper function for prediction
    def predict_image(model, img_path, transform, device, label_encoder):
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Get top prediction
        top_prob, top_class = torch.max(probabilities, 0)
        predicted_class = label_encoder.inverse_transform([top_class.item()])[0]
        confidence = float(top_prob.item()) * 100

        return predicted_class, confidence, probabilities.cpu().numpy()

    # Find test images
    test_images = []
    for disease_folder in os.listdir(sample_images_dir):
        disease_folder_path = os.path.join(sample_images_dir, disease_folder)
        if not os.path.isdir(disease_folder_path):
            continue

        img_files = [f for f in os.listdir(disease_folder_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if img_files:
            # Take a random image from this disease category
            selected_img = os.path.join(disease_folder_path, random.choice(img_files))
            test_images.append((selected_img, disease_folder))

    # Randomly select images for demonstration
    num_images = min(6, len(test_images))  # Increased from 4 to 6 images
    selected_test_images = random.sample(test_images, num_images) if len(test_images) > num_images else test_images

    # Set up the figure with GridSpec for better layout control
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(3, 2, figure=fig)

    # Add a stylish title with improved formatting
    fig.suptitle("🌿 Plant Disease Diagnosis & Treatment Recommendations", 
                 fontsize=28, fontweight='bold', y=0.98, 
                 bbox=dict(facecolor='#e8f4ea', edgecolor='green', boxstyle='round,pad=0.5'))

    # Add a subtitle with system info
    fig.text(0.5, 0.94, f"Running on: {device} | Model: PlantDiseaseModel | Classes: {num_classes}", 
             ha='center', fontsize=14, fontstyle='italic', color='#555555')

    # Color palette for recommendations
    treatment_colors = {
        'healthy': '#e8f4ea',  # Light green
        'disease': '#f9e8ea'   # Light red
    }

    # Add a legend for confidence
    cmap = plt.cm.RdYlGn
    confidence_gradient = np.linspace(0, 1, 100)
    confidence_bar = np.vstack((confidence_gradient, confidence_gradient))

    # Add confidence colorbar at the bottom
    cax = fig.add_axes([0.3, 0.05, 0.4, 0.02])
    cb = plt.colorbar(plt.imshow(confidence_bar, cmap=cmap), cax=cax, orientation='horizontal')
    cb.set_label('Prediction Confidence', fontsize=14)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cb.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    # Create grid layout based on number of images
    rows = 2 if num_images <= 4 else 3
    cols = 2

    # Process each selected image
    all_predictions = []  # Store prediction results
    for i, (img_path, true_label) in enumerate(selected_test_images):
        if i >= rows * cols:
            break

        # Calculate grid position
        row = i // cols
        col = i % cols

        # Make prediction
        predicted_class, confidence, probabilities = predict_image(
            model, img_path, transform, device, label_encoder)

        # Store prediction result
        all_predictions.append((os.path.basename(img_path), true_label, predicted_class, confidence))

        # Create subplot with better positioning
        ax = fig.add_subplot(gs[row, col])

        # Load and display image
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
        ax.axis('off')

        # Determine if prediction is correct
        is_correct = predicted_class == true_label

        # Format disease name for display
        display_pred = predicted_class.replace('_', ' ')
        display_true = true_label.replace('_', ' ')

        # Apply color based on confidence
        title_color = cmap(confidence/100)

        # Create a styled title box
        title_box = dict(
            boxstyle='round,pad=0.5',
            facecolor=cmap(confidence/100),
            alpha=0.8,
            edgecolor='gray'
        )

        # Set title with prediction info in a box
        ax.set_title(f"Prediction: {display_pred}\nConfidence: {confidence:.1f}%", 
                    fontsize=16, fontweight='bold', color='white',
                    bbox=title_box)

        # Add actual label in smaller text
        ax.text(0.5, -0.05, f"Actual: {display_true}", 
                transform=ax.transAxes, ha='center', fontsize=14,
                color='black' if is_correct else 'darkred',
                fontweight='bold' if not is_correct else 'normal')

        # Get treatment recommendations
        is_healthy = "healthy" in predicted_class.lower()
        if is_healthy:
            recommendations = default_healthy_practices
            recommendation_title = "Healthy Plant Care:"
            box_color = treatment_colors['healthy']
        else:
            recommendations = treatment_recommendations.get(
                predicted_class, ["No specific recommendations available"])
            recommendation_title = "Treatment Recommendations:"
            box_color = treatment_colors['disease']

        # Create a styled box for recommendations
        rec_box_props = dict(
            boxstyle='round,pad=0.6',
            facecolor=box_color,
            alpha=0.85,
            edgecolor='gray'
        )

        # Add disease severity indicator
        if not is_healthy:
            severity = "High" if confidence > 85 else "Medium" if confidence > 65 else "Low"
            severity_color = "red" if severity == "High" else "orange" if severity == "Medium" else "green"
            severity_text = f"Severity: {severity}"
        else:
            severity_text = "Status: Healthy"
            severity_color = "green"

        # Build recommendation text with formatting
        recommendation_text = f"{recommendation_title}\n"
        for rec in recommendations:
            recommendation_text += f"• {rec}\n"

        recommendation_text += f"\n{severity_text}"

        # Place recommendations in a better position
        plt.figtext(0.5 + col * 0.5 - 0.48, 
                   0.9 - row * 0.33 - 0.13,
                   recommendation_text, 
                   fontsize=14,
                   color='black',
                   bbox=rec_box_props,
                   verticalalignment='top')

        # Add severity indicator dot
        plt.figtext(0.5 + col * 0.5 - 0.15, 
                   0.9 - row * 0.33 - 0.33,
                   "●", 
                   fontsize=30,
                   color=severity_color,
                   ha='right')

        # Add top 3 probable diseases as small text (if not healthy)
        if not is_healthy and len(class_names) > 1:
            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_classes = [label_encoder.inverse_transform([idx])[0].replace('_', ' ') for idx in top_indices]
            top_probs = [probabilities[idx] * 100 for idx in top_indices]

            # Format alternatives text
            alt_text = "Alternative diagnoses:\n"
            for j, (cls, prob) in enumerate(zip(top_classes, top_probs)):
                if j == 0:  # Skip the top prediction (already shown)
                    continue
                alt_text += f"{cls}: {prob:.1f}%\n"

            # Add alternatives in small text
            plt.figtext(0.5 + col * 0.5 - 0.48, 
                       0.9 - row * 0.33 - 0.3,
                       alt_text, 
                       fontsize=10,
                       color='#555555',
                       verticalalignment='top')

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    # Add a footer with additional information
    footer_text = (
        "Note: This is an AI-assisted diagnosis tool and should be used as a guide only. "
        "For conclusive diagnosis, consult with a professional plant pathologist."
    )
    fig.text(0.5, 0.01, footer_text, ha='center', fontsize=12, fontstyle='italic')

    plt.show()

    # Return results summary for further use if needed
    results = {
        "images_analyzed": len(selected_test_images),
        "predictions": all_predictions,
        "model_device": str(device)
    }

    return results

print("\n🌿 Running interactive disease diagnosis with treatment recommendations...")
results = interactive_disease_diagnosis(
    model_path=model_path,
    label_encoder_path="label_encoder.pkl",
    transform_path="inference_transform.pkl",
    sample_images_dir=data_dir
)

# Print a summary of results
print(f"\nAnalysis complete! Examined {results['images_analyzed']} plant images")
print(f"Model running on: {results['model_device']}")
print("\nSummary of diagnoses:")
for img, true, pred, conf in results['predictions']:
    match = "✓" if true.replace("_", " ") == pred.replace("_", " ") else "✗"
    print(f"- {img}: {pred.replace('_', ' ')} ({conf:.1f}%) {match}")

