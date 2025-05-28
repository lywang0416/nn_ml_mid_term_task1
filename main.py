import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import json

def save_results_to_file(results, filename="training_results.json"):
    """将结果保存到 JSON 文件"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def load_results_from_file(filename="training_results.json"):
    """从文件加载历史结果"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record history
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())
                
                # deep copy the model if best
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    history = {
        'train_loss': train_loss_history,
        'train_acc': train_acc_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history
    }
    
    return model, history

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    # Iterate over data
    for inputs, labels in tqdm(dataloader, desc='Testing'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # forward
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
        
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    test_loss = running_loss / len(dataloader.dataset)
    test_acc = running_corrects.double() / len(dataloader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    return test_loss, test_acc.item()

def plot_training_history(history, title):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    save_path = f'{title.replace(" ", "_")}_training_history.png'
    plt.savefig(save_path)
    plt.close()
    print(f"训练曲线已保存到: {save_path}")

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement
    model_ft = None
    
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        model_ft = models.resnet18(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "alexnet":
        weights = models.AlexNet_Weights.DEFAULT if use_pretrained else None
        model_ft = models.alexnet(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    else:
        print(f"Invalid model name: {model_name}")
        return None
    
    return model_ft

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    data_dir = './caltech_101_data/101_ObjectCategories'
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters to test
    models_to_test = ["resnet18", "alexnet"]
    learning_rates = [0.001, 0.0001]
    num_epochs_list = [15,25]
    
    # 初始化结果文件和缓存
    results_file = "training_results.json"
    if os.path.exists(results_file):
        # 如果文件已存在，加载历史结果
        results = load_results_from_file(results_file)
    else:
        results = []
        
    # 1. Using pretrained models with feature extraction (only training the last layer)
    existing_results = load_results_from_file(results_file)
    for model_name in models_to_test:
        for lr in learning_rates:
            for num_epochs in num_epochs_list:
                # === 检查是否已训练过 ===
                already_trained = any(
                    r['model'] == model_name 
                    and r['mode'] == 'pretrained_feature_extraction'
                    and r['lr'] == lr 
                    and r['epochs'] == num_epochs
                    for r in existing_results
                )
                if already_trained:
                    print(f"Skipping {model_name} (lr={lr}, epochs={num_epochs}): already trained")
                    continue
                # === 检查结束 ===
                
                print(f"\n{'='*80}")
                print(f"Training {model_name} with pretrained weights, feature extraction, lr={lr}, epochs={num_epochs}")
                print(f"{'='*80}")
                
                # Initialize model
                model_ft = initialize_model(model_name, num_classes=101, feature_extract=True, use_pretrained=True)
                model_ft = model_ft.to(device)
                
                # Setup training
                params_to_update = [p for p in model_ft.parameters() if p.requires_grad]
                optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=0.9)
                criterion = nn.CrossEntropyLoss()
                
                # Train model
                model_ft, history = train_model(model_ft, dataloaders, criterion, optimizer_ft, 
                                               scheduler=None, device=device, num_epochs=num_epochs)
                # Save model
                save_name = f"{model_name}_pretrained_feature_lr{lr}_epochs{num_epochs}.pth"
                torch.save(model_ft.state_dict(), save_name)
                print(f"模型权重已保存到: {save_name}")
                
                # Evaluate on test set
                test_loss, test_acc = evaluate_model(model_ft, test_loader, criterion, device)
                
                # Plot results
                plot_title = f"{model_name} Pretrained Feature Extraction (lr={lr}, epochs={num_epochs})"
                plot_training_history(history, plot_title)
                
                # Save results
                results.append({
                    'model': model_name,
                    'mode': 'pretrained_feature_extraction',
                    'lr': lr,
                    'epochs': num_epochs,
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                    'train_history': history
                })
                save_results_to_file(results, results_file)
    
    # 2. Fine-tuning pretrained models (training all layers with different learning rates)
    for model_name in models_to_test:
        for lr in learning_rates:
            for num_epochs in num_epochs_list:
                # === 检查是否已训练过 ===
                already_trained = any(
                    r['model'] == model_name 
                    and r['mode'] == 'pretrained_fine_tuning'
                    and r['lr'] == lr 
                    and r['epochs'] == num_epochs
                    for r in existing_results
                )
                if already_trained:
                    print(f"Skipping {model_name} (lr={lr}, epochs={num_epochs}): already trained")
                    continue
                # === 检查结束 ===
                
                print(f"\n{'='*80}")
                print(f"Training {model_name} with pretrained weights, fine-tuning, lr={lr}, epochs={num_epochs}")
                print(f"{'='*80}")
                
                # Initialize model
                model_ft = initialize_model(model_name, num_classes=101, feature_extract=False, use_pretrained=True)
                model_ft = model_ft.to(device)
                
                # Set up different learning rates for different layers
                # Lower learning rate for pretrained parameters, higher for the new layers
                if model_name == "resnet18":
                    optimizer_ft = optim.SGD([
                        {'params': model_ft.layer1.parameters(), 'lr': lr/10},
                        {'params': model_ft.layer2.parameters(), 'lr': lr/10},
                        {'params': model_ft.layer3.parameters(), 'lr': lr/10},
                        {'params': model_ft.layer4.parameters(), 'lr': lr/5},
                        {'params': model_ft.fc.parameters(), 'lr': lr}
                    ], momentum=0.9)
                elif model_name == "alexnet":
                    optimizer_ft = optim.SGD([
                        {'params': model_ft.features.parameters(), 'lr': lr/10},
                        {'params': model_ft.classifier[:-1].parameters(), 'lr': lr/5},
                        {'params': model_ft.classifier[-1].parameters(), 'lr': lr}
                    ], momentum=0.9)
                
                criterion = nn.CrossEntropyLoss()
                
                # Train model
                model_ft, history = train_model(model_ft, dataloaders, criterion, optimizer_ft, 
                                               scheduler=None, device=device, num_epochs=num_epochs)
                
                # Save model
                save_name = f"{model_name}_pretrained_finetune_lr{lr}_epochs{num_epochs}.pth"
                torch.save(model_ft.state_dict(), save_name)
                print(f"模型权重已保存到: {save_name}")

                # Evaluate on test set
                test_loss, test_acc = evaluate_model(model_ft, test_loader, criterion, device)
                
                # Plot results
                plot_title = f"{model_name} Pretrained Fine-tuning (lr={lr}, epochs={num_epochs})"
                plot_training_history(history, plot_title)
                
                # Save results
                results.append({
                    'model': model_name,
                    'mode': 'pretrained_fine_tuning',
                    'lr': lr,
                    'epochs': num_epochs,
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                    'train_history': history
                })
                save_results_to_file(results, results_file)
    
    # 3. Training from scratch for comparison
    for model_name in models_to_test:
        # === 检查是否已训练过 ===
        already_trained = any(
            r['model'] == model_name 
            and r['mode'] == 'from_scratch'
            for r in existing_results
        )
        if already_trained:
            print(f"Skipping {model_name} from scratch: already trained")
            continue
        # === 检查结束 ===
        
        print(f"\n{'='*80}")
        print(f"Training {model_name} from scratch, lr=0.001, epochs=25")
        print(f"{'='*80}")
        
        # Initialize model with random weights
        model_scratch = initialize_model(model_name, num_classes=101, feature_extract=False, use_pretrained=False)
        model_scratch = model_scratch.to(device)
        
        # Setup training
        optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        model_scratch, history = train_model(model_scratch, dataloaders, criterion, optimizer_scratch, 
                                           scheduler=None, device=device, num_epochs=25)

        # Save model
        save_name = f"{model_name}_from_scratch.pth"
        torch.save(model_scratch.state_dict(), save_name)
        print(f"模型权重已保存到: {save_name}")

        # Evaluate on test set
        test_loss, test_acc = evaluate_model(model_scratch, test_loader, criterion, device)
        
        # Plot results
        plot_title = f"{model_name} Trained From Scratch" 
        plot_training_history(history, plot_title)
        
        # Save results
        results.append({
            'model': model_name,
            'mode': 'from_scratch',
            'lr': 0.001,
            'epochs': 25,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'train_history': history
        })
        save_results_to_file(results, results_file)
    
    # Print comparison of results
    print("\n\nResults Summary:")
    print("=" * 100)
    print(f"{'Model':<12} {'Mode':<30} {'Learning Rate':<15} {'Epochs':<8} {'Test Accuracy':<15} {'Test Loss':<10}")
    print("=" * 100)

    # Load results
    saved_results = load_results_from_file(results_file)
    for result in saved_results:
        print(
            f"{result['model']:<12} "
            f"{result['mode']:<30} "
            f"{result['lr']:<15} "
            f"{result['epochs']:<8} "
            f"{result['test_acc']:<15.4f} "
            f"{result['test_loss']:<10.4f}"
        )
if __name__ == "__main__":
    main()
