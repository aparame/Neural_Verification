import os
import sys
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd

class ProcessedImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, csv_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        self.data['timestamp'] = self.data['timestamp'].astype(str).str.replace("M_", "")
        self.label_map = dict(zip(self.data['timestamp'], self.data['steering']))
        self.image_filenames = sorted([f for f in os.listdir(image_folder) 
                                     if os.path.isfile(os.path.join(image_folder, f))])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image = Image.open(os.path.join(self.image_folder, img_name)).convert('L')
        timestamp = img_name.replace("RGB_", "").replace(".png", "")
        label = self.label_map.get(timestamp, -1)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

class NvidiaNet(nn.Module):
    def __init__(self):
        super(NvidiaNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(36)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(64)
        self._to_linear = self._get_conv_output((1, 80, 64))
        self.fc1 = nn.Linear(self._to_linear, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)
        self.dropout = nn.Dropout(p=0.3)
    
    def _get_conv_output(self, shape):
        x = torch.rand(1, *shape)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.conv4(x))
        x = self.bn5(self.conv5(x))
        return int(torch.prod(torch.tensor(x.size()[1:])))
    
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.conv4(x))
        x = self.bn5(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class ResNet18Regressor(nn.Module):
    def __init__(self,pretrained=True):
        super(ResNet18Regressor, self).__init__()
        # Load the pre-trained ResNet18 model
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the first convolutional layer to accept grayscale images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the fully connected layer to output a single value
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
    
    def forward(self, x):
        return self.resnet(x)

def train_model(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                         lr=config.getfloat('TRAINING', 'learning_rate'),
                         weight_decay=config.getfloat('TRAINING', 'weight_decay'))
    
    best_val_loss = float('inf')
    for epoch in range(config.getint('TRAINING', 'num_epochs')):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs.squeeze(), labels).item()
        
        avg_train = train_loss/len(train_loader)
        avg_val = val_loss/len(val_loader)
        print(f'Epoch {epoch+1}/{config.getint("TRAINING", "num_epochs")}')
        print(f'Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}')
        
        # if avg_val < best_val_loss:
        #     best_val_loss = avg_val
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': best_val_loss,
        #     }, os.path.join(config['PATHS']['save_dir'], f'best_{model.__class__.__name__}.pt'))
    
    return model

def main():
    config = configparser.ConfigParser()
    config.read('config_NNC.ini')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    dataset = ProcessedImageDataset(
        config['PATHS']['image_folder'],
        config['PATHS']['csv_file'],
        transform=transform
    )
    
    train_size = int(config.getfloat('TRAINING', 'train_ratio') * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=config.getint('TRAINING', 'batch_size'),
                            shuffle=True,
                            num_workers=config.getint('TRAINING', 'num_workers'))
    
    test_loader = DataLoader(test_dataset,
                           batch_size=config.getint('TRAINING', 'batch_size'),
                           shuffle=False,
                           num_workers=config.getint('TRAINING', 'num_workers'))
    
    models_to_train = {
        'NvidiaNet': NvidiaNet(),
        'ResNet18': ResNet18Regressor(
            pretrained=config.getboolean('RESNET18', 'pretrained')
        )
    }
    
    os.makedirs(config['PATHS']['save_dir'], exist_ok=True)
    
    for name, model in models_to_train.items():
        print(f'\n{"="*30}')
        print(f'Training {name}')
        print(f'{"="*30}\n')
        
        # Train the model
        trained_model = train_model(model, train_loader, test_loader, config)
        
        # Create model save paths
        model_dir = os.path.join(config['PATHS']['save_dir'], name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save PyTorch model
        torch_path = os.path.join(model_dir, f'{name}.pt')
        torch.save(trained_model.state_dict(), torch_path)
        print(f'Saved PyTorch model to {torch_path}')
        
        # Save ONNX model
        onnx_path = os.path.join(model_dir, f'{name}.onnx')
        dummy_input = torch.randn(1, 1, 80, 64).to(device)  # Adjust input size as needed
        torch.onnx.export(
            trained_model,
            dummy_input,
            onnx_path,
            opset_version=11
        )
        print(f'Saved ONNX model to {onnx_path}')
        
        trained_model.eval()
        test_mse = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = trained_model(images)  # Use trained_model instead of model
                test_mse += torch.mean((outputs.squeeze() - labels) ** 2)  # Squeeze outputs

        test_mse /= len(test_loader)
        print(f"Test MSE: {test_mse.item():.4f}")

if __name__ == '__main__':
    main()