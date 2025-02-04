import os
import configparser
import torch
import torch.nn as nn
from torchvision import models

# Define model architectures
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
    def __init__(self):
        super(ResNet18Regressor, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
    
    def forward(self, x):
        return self.resnet(x)

class VAE(nn.Module):
    def __init__(self, latent_dim, flattened_size, flattened_size_decoder, n_gaussians=3):
        super(VAE, self).__init__()
        self.n_gaussians = n_gaussians
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.25)
        )

        self.flattened_size = flattened_size
        self.fc_mu = nn.ModuleList([nn.Linear(flattened_size, latent_dim) for _ in range(n_gaussians)])
        self.fc_log_var = nn.ModuleList([nn.Linear(flattened_size, latent_dim) for _ in range(n_gaussians)])
        self.fc_weights = nn.Linear(flattened_size, n_gaussians)
        
        self.decoder = Decoder(latent_dim, flattened_size_decoder)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        # Add numerical stability
        x = torch.clamp(x, min=-1e6, max=1e6)  # Prevent extreme values
        
        mu = [mu_layer(x) for mu_layer in self.fc_mu]
        log_var = [log_var_layer(x) for log_var_layer in self.fc_log_var]
        weights = F.softmax(self.fc_weights(x), dim=-1)  # Compute softmax over Gaussian components

        # Stabilize weight logits
        weight_logits = self.fc_weights(x)
        weight_logits = torch.clamp(weight_logits, min=-50, max=50)  # Prevent overflow
        weights = F.softmax(weight_logits, dim=-1)
        
        # Add epsilon to avoid zero probabilities
        weights = weights + 1e-8
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        return mu, log_var, weights

    def reparameterize(self, mus, log_vars, weights):
        
        # Convert to double precision for sampling
        weights = weights.double()
        
        batch_size = mus[0].size(0)
        gaussian_idx = torch.multinomial(
            weights, 
            1,
            replacement=True  # Ensure valid sampling even with precision issues
        ).squeeze()
        
        # Convert back to original precision
        mu = torch.stack(mus)[gaussian_idx, torch.arange(batch_size)].float() 
        log_var = torch.stack(log_vars)[gaussian_idx, torch.arange(batch_size)].float()
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var, weights = self.encode(x)
        z = self.reparameterize(mu, log_var, weights)
        return self.decode(z), mu, log_var, weights

class Decoder(nn.Module):
    def __init__(self, latent_dim, flattened_size):
        super(Decoder, self).__init__()
        self.input_layer = nn.Linear(latent_dim, flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.input_layer(z)
        x = z.view(z.size(0), 256, 5, 4)
        x = self.decoder(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, decoder, controller):
        super(CombinedModel, self).__init__()
        self.decoder = decoder
        self.controller = controller
    
    def forward(self, z):
        x = self.decoder(z)
        output = self.controller(x)
        return output

def load_components(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load controllers
    nvidia_net = NvidiaNet()
    nvidia_net.load_state_dict(torch.load(config['PATHS']['nvidia_net_path']))
    nvidia_net.to(device)
    
    resnet18 = ResNet18Regressor()
    resnet18.load_state_dict(torch.load(config['PATHS']['resnet18_path']))
    resnet18.to(device)
    
    return nvidia_net, resnet18, device

def process_vaes(config, controllers, device):
    save_dir = config['PATHS']['save_dir']
    save_combine_dir = config['PATHS']['save_combine_dir']
    os.makedirs(save_combine_dir, exist_ok=True)

    # Get all VAE checkpoints
    vae_files = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    
    for vae_file in vae_files:
        # Load VAE
        vae_path = os.path.join(save_dir, vae_file)
        vae = VAE(
        latent_dim=eval(config.get('Model', 'latent_dim')),
        flattened_size=eval(config.get('Model', 'encoder_flatten')),
        flattened_size_decoder=eval(config.get('Model', 'decoder_flatten')),
        n_gaussians=config.getint('Model', 'n_gaussians')
        )
        vae.load_state_dict(torch.load(vae_path))
        vae.to(device)
        
        # Extract decoder
        decoder = vae.decoder
        
        # Create base name from VAE filename
        base_name = os.path.splitext(vae_file)[0]
        
        # Create and save combined models
        for ctrl_name, controller in controllers.items():
            combined = CombinedModel(decoder, controller)
            combined.eval()
            
            # Export combined model
            model_name = f"{ctrl_name}_{base_name}"
            export_combined(combined, model_name, save_combine_dir, device)

def export_combined(model, model_name, save_combine_dir, device):
    model.eval()
    example_input = torch.randn(1, eval(config.get('Model', 'latent_dim'))).to(device)
    
    # Export ONNX
    onnx_path = os.path.join(save_combine_dir, f"{model_name}.onnx")
    torch.onnx.export(model, example_input, onnx_path,
                     opset_version=11, input_names=['input'],
                     output_names=['output'], 
                     dynamic_axes={'input': {0: 'batch_size'},
                                   'output': {0: 'batch_size'}})
    
    # Export TorchScript
    pt_path = os.path.join(save_combine_dir, f"{model_name}.pt")
    traced = torch.jit.trace(model, example_input)
    torch.jit.save(traced, pt_path)
    
    print(f"Exported {model_name} to:\n- {onnx_path}\n- {pt_path}\n")

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('NNC_config.ini')
    
    # Load controllers
    nvidia_net, resnet18, device = load_components(config)
    controllers = {
        'NvidiaNet': nvidia_net,
        'ResNet18': resnet18
    }
    
    # Process all VAEs and create combined models
    process_vaes(config, controllers, device)