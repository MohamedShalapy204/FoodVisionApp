
import torch
import torchvision

from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_effnet_model(num_classes: int=3,
                        seed: int=42):
  weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
  transforms = weights.transforms()
  model = torchvision.models.efficientnet_b0(weights=weights).to(device)

  for param in model.parameters():
    param.requires_grid = False
  
  torch.manual_seed(seed)
  model.classifier = nn.Sequential(
      nn.Dropout(p=0.3, inplace=True),
      nn.Linear(in_features=1280, out_features=num_classes)
  )
  return model, transforms
