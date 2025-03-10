import torch
import torch.fft
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import torch.nn as nn

class NormalizationLayer:
    """Factory class for creating normalization layers"""
    @staticmethod
    def get_norm_layer(norm_type, num_channels, **kwargs):
        if norm_type == "batch":
            return nn.BatchNorm2d(num_channels)
        elif norm_type == "instance":
            return nn.InstanceNorm2d(num_channels, affine=True)
        elif norm_type == "layer":
            return nn.GroupNorm(1, num_channels)  # Layer norm is equivalent to GroupNorm with num_groups=1
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")
            

class FFParser(torch.nn.Module):
    def __init__(self, channels, fmap_size, remove_dc=False):
        super(FFParser, self).__init__()
        # Learnable attention map generator
        self.attn_map = torch.nn.Parameter(torch.ones(channels, fmap_size, fmap_size))
        #self.attn_map = torch.nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.center_x = fmap_size // 2
        self.center_y = fmap_size // 2
        self.remove_dc = remove_dc
    
    def forward(self, x):
        # Apply FFT

        x = x.type(torch.float32)
        x_fft = torch.fft.fft2(x, dim=(-2, -1))   # FFT along spatial dimensions
        x_fft = torch.fft.fftshift(x_fft)       # Shift zero-frequency component to the center
        
        # # Inside the training loop
        #if self.remove_dc:
        dc_component = x_fft[:, :, self.center_x, self.center_y].clone()
        x_fft[:, :, self.center_x, self.center_y] = 0
        #print(dc_component)
        #x_fft = x_fft / (torch.abs(x_fft).max() + 1e-6)
        #print(center_x, center_y, x_fft[0,0])
        
        # Generate and apply attention map
        magnitude_before = torch.abs(x_fft).cpu()  
        
        
        x_fft = x_fft * self.attn_map
        magnitude_after = torch.abs(x_fft).cpu()
        x_fft[:, :, self.center_x, self.center_y] = dc_component
        
        # Apply inverse FFT
        x_fft = torch.fft.ifftshift(x_fft)     
        x_out = torch.fft.ifft2(x_fft, dim=(-2, -1)).type(torch.float32)  
        x_out = x_out.real.type(torch.float32)                       

        attention_map = self.attn_map.cpu()
        #magnitude_before = magnitude_after = attention_map = self.attn_map.cpu()
        
        return x_out , magnitude_before, magnitude_after, attention_map

    @staticmethod
    def visualize(x, x_out, magnitude_before, magnitude_after, attention_map, layer_name):
        plt.figure(figsize=(40, 6))
        plt.subplot(1, 6, 1)
        plt.title(f"FFT Magnitude Before - {layer_name}")
        plt.imshow(magnitude_before[0, 10].detach().numpy())

        plt.subplot(1, 6, 2)
        plt.title(f"FFT Magnitude After - {layer_name}")
        plt.imshow(magnitude_after[0, 10].detach().numpy())

        plt.subplot(1, 6, 4)
        plt.title(f"Image before - {layer_name}")
        plt.imshow(x[0, 0].cpu().detach().numpy(), cmap='viridis')
        plt.colorbar()

        plt.subplot(1, 6, 3)
        plt.title(f"Attention map - {layer_name}")
        plt.imshow(attention_map[10].detach().numpy(), cmap='viridis')
        plt.colorbar()
        
        plt.subplot(1, 6, 5)
        plt.title(f"Image after - {layer_name}")
        plt.imshow(x_out[0, 10].cpu().detach().numpy(), cmap='viridis')
        plt.colorbar()

        plt.subplot(1, 6, 6)
        plt.title(f"after - before - {layer_name}")
        plt.imshow(x_out[0, 10].cpu().detach().numpy() - x[0, 0].cpu().detach().numpy(), cmap='viridis')
        plt.colorbar()
        
        
        plt.tight_layout()
        plt.savefig(f"./plots/sample_{layer_name}.jpg", dpi=500)
        #plt.close()


class NormalizationLayer:
    """Factory class for creating normalization layers"""
    @staticmethod
    def get_norm_layer(norm_type, num_channels, **kwargs):
        if norm_type == "batch":
            return nn.BatchNorm2d(num_channels)
        elif norm_type == "instance":
            return nn.InstanceNorm2d(num_channels, affine=True)
        elif norm_type == "layer":
            return nn.GroupNorm(1, num_channels)  # Layer norm is equivalent to GroupNorm with num_groups=1
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

class ConfigurableBottleneck(nn.Module):
    """Modified Bottleneck block with configurable normalization"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_type="batch"):
        super(ConfigurableBottleneck, self).__init__()
        width = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = NormalizationLayer.get_norm_layer(norm_type, width)
        
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = NormalizationLayer.get_norm_layer(norm_type, width)
        
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = NormalizationLayer.get_norm_layer(norm_type, out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class FFResNet50(nn.Module):
    def __init__(self, norm_type="batch"):
        super(FFResNet50, self).__init__()
        
        # Initial layers with configurable normalization
        self.initial = torch.nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            NormalizationLayer.get_norm_layer(norm_type, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),)
        
        # FF-Parser after initial layer
        self.ff_parser_1 = FFParser(channels=64, fmap_size=128, remove_dc=True)
        
        # Create residual stages with configurable normalization
        self.layer1 = self._make_layer(64, 256, 3, norm_type=norm_type)
        self.ff_parser_2 = FFParser(channels=256, fmap_size=128, remove_dc=True)
        
        self.layer2 = self._make_layer(256, 512, 4, stride=2, norm_type=norm_type)
        self.ff_parser_3 = FFParser(channels=512, fmap_size=64)
        
        self.layer3 = self._make_layer(512, 1024, 6, stride=2, norm_type=norm_type)
        self.ff_parser_4 = FFParser(channels=1024, fmap_size=32)
        
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2, norm_type=norm_type)
        self.ff_parser_5 = FFParser(channels=2048, fmap_size=16)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 2)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, norm_type="batch"):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                NormalizationLayer.get_norm_layer(norm_type, out_channels)
            )
        
        layers = []
        layers.append(ConfigurableBottleneck(in_channels, out_channels, stride, downsample, norm_type))
        
        for _ in range(1, blocks):
            layers.append(ConfigurableBottleneck(out_channels, out_channels, norm_type=norm_type))
            
        return nn.Sequential(*layers)

    def forward(self, x, visualize=False):
        # Initial layers
        x1 = self.initial(x)
        
        x, magnitude_before, magnitude_after, attention_map = self.ff_parser_1(x1)
        if visualize:
            self.ff_parser_1.visualize(x1, x, magnitude_before, magnitude_after, attention_map, "FFParser 1")
        
        # Residual stages with FF-Parser
        x1 = self.layer1(x)
        x, magnitude_before, magnitude_after, attention_map = self.ff_parser_2(x1)
        if visualize:
            self.ff_parser_2.visualize(x1, x, magnitude_before, magnitude_after, attention_map, "FFParser 2")
        
        x1 = self.layer2(x)
        x, magnitude_before, magnitude_after, attention_map = self.ff_parser_3(x1)
        if visualize:
            self.ff_parser_3.visualize(x1, x, magnitude_before, magnitude_after, attention_map, "FFParser 3")
        
        x1 = self.layer3(x)
        x, magnitude_before, magnitude_after, attention_map = self.ff_parser_4(x1)
        if visualize:
            self.ff_parser_4.visualize(x1, x, magnitude_before, magnitude_after, attention_map, "FFParser 4")
        
        x1 = self.layer4(x)
        x, magnitude_before, magnitude_after, attention_map = self.ff_parser_5(x1)
        if visualize:
            self.ff_parser_5.visualize(x1, x, magnitude_before, magnitude_after, attention_map, "FFParser 5")
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ConfigurableBasicBlock(nn.Module):
    """Modified BasicBlock with configurable normalization"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_type="batch"):
        super(ConfigurableBasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormalizationLayer.get_norm_layer(norm_type, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = NormalizationLayer.get_norm_layer(norm_type, out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out
        
class FFResNet34(nn.Module):
    def __init__(self, norm_type="batch"):
        super(FFResNet34, self).__init__()
        
        # Initial layers with configurable normalization
        self.initial = torch.nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            NormalizationLayer.get_norm_layer(norm_type, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # FF-Parser after initial layer
        self.ff_parser_1 = FFParser(channels=64, fmap_size=128, remove_dc=True)
        
        # Create residual stages with configurable normalization
        self.layer1 = self._make_layer(64, 64, 3, norm_type=norm_type)
        self.ff_parser_2 = FFParser(channels=64, fmap_size=128, remove_dc=True)
        
        self.layer2 = self._make_layer(64, 128, 4, stride=2, norm_type=norm_type)
        self.ff_parser_3 = FFParser(channels=128, fmap_size=64)
        
        self.layer3 = self._make_layer(128, 256, 6, stride=2, norm_type=norm_type)
        self.ff_parser_4 = FFParser(channels=256, fmap_size=32)
        
        self.layer4 = self._make_layer(256, 512, 3, stride=2, norm_type=norm_type)
        self.ff_parser_5 = FFParser(channels=512, fmap_size=16)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, norm_type="batch"):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                NormalizationLayer.get_norm_layer(norm_type, out_channels)
            )
        
        layers = []
        layers.append(ConfigurableBasicBlock(in_channels, out_channels, stride, downsample, norm_type))
        
        for _ in range(1, blocks):
            layers.append(ConfigurableBasicBlock(out_channels, out_channels, norm_type=norm_type))
            
        return nn.Sequential(*layers)

    def forward(self, x, visualize=False):
        # Initial layers
        x1 = self.initial(x)
        
        x, magnitude_before, magnitude_after, attention_map = self.ff_parser_1(x1)
        if visualize:
            self.ff_parser_1.visualize(x1, x, magnitude_before, magnitude_after, attention_map, "FFParser 1")
        
        # Residual stages with FF-Parser
        x1 = self.layer1(x)
        x, magnitude_before, magnitude_after, attention_map = self.ff_parser_2(x1)
        if visualize:
            self.ff_parser_2.visualize(x1, x, magnitude_before, magnitude_after, attention_map, "FFParser 2")
        
        x1 = self.layer2(x)
        x, magnitude_before, magnitude_after, attention_map = self.ff_parser_3(x1)
        if visualize:
            self.ff_parser_3.visualize(x1, x, magnitude_before, magnitude_after, attention_map, "FFParser 3")
        
        x1 = self.layer3(x)
        x, magnitude_before, magnitude_after, attention_map = self.ff_parser_4(x1)
        if visualize:
            self.ff_parser_4.visualize(x1, x, magnitude_before, magnitude_after, attention_map, "FFParser 4")
        
        x1 = self.layer4(x)
        x, magnitude_before, magnitude_after, attention_map = self.ff_parser_5(x1)
        if visualize:
            self.ff_parser_5.visualize(x1, x, magnitude_before, magnitude_after, attention_map, "FFParser 5")
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        

class oldFFResNet50(torch.nn.Module):
    def __init__(self):
        super(oldFFResNet50, self).__init__()
        # Load the base ResNet50
        base_model = resnet50()

        # Initial layers (Conv1 + MaxPool)
        self.initial = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )

        # FF-Parser after initial layer
        self.ff_parser_1 = FFParser(channels=64, fmap_size=128)
        
        # Residual stages (layer1 to layer4)
        self.layer1 = base_model.layer1
        self.ff_parser_2 = FFParser(channels=256, fmap_size=128)

        self.layer2 = base_model.layer2
        self.ff_parser_3 = FFParser(channels=512, fmap_size=64)

        self.layer3 = base_model.layer3
        self.ff_parser_4 = FFParser(channels=1024, fmap_size=32)

        self.layer4 = base_model.layer4
        self.ff_parser_5 = FFParser(channels=2048, fmap_size=16)
        self.avgpool = base_model.avgpool
        self.fc = torch.nn.Linear(2048, 2)

    def forward(self, x, visualize=False):
        # Initial layers
        x1 = self.initial(x)
        x, magnitude_before, magnitude_after, attention_map = self.ff_parser_1(x1)
        if visualize:
            self.ff_parser_1.visualize(x1, x, magnitude_before, magnitude_after, attention_map, "FFParser 1")

        # Residual stages with FF-Parser
        x1 = self.layer1(x)

        x, magnitude_before, magnitude_after, attention_map = self.ff_parser_2(x1)
        if visualize:
            self.ff_parser_2.visualize(x1, x, magnitude_before, magnitude_after, attention_map, "FFParser 2")

        x1 = self.layer2(x)
        x, magnitude_before, magnitude_after, attention_map  = self.ff_parser_3(x1)
        if visualize:
            self.ff_parser_3.visualize(x1, x, magnitude_before, magnitude_after, attention_map, "FFParser 3")

        x1 = self.layer3(x)
        x, magnitude_before, magnitude_after, attention_map = self.ff_parser_4(x1)
        if visualize:
            self.ff_parser_4.visualize(x1, x, magnitude_before, magnitude_after, attention_map, "FFParser 4")

        x1 = self.layer4(x)
        x, magnitude_before, magnitude_after, attention_map  = self.ff_parser_5(x1)
        if visualize:
            self.ff_parser_5.visualize(x1, x, magnitude_before, magnitude_after, attention_map, "FFParser 5")
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x