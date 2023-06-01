from torchvision import models
import copy 
import torch

resnet18_pretrained = models.resnet18(pretrained=False)
copy_state_dict = copy.deepcopy(resnet18_pretrained.state_dict())
for key in resnet18_pretrained.state_dict().keys():
    if key == "fc.bias":
        copy_state_dict[key] = torch.zeros_like(resnet18_pretrained.state_dict()[key])

resnet18_pretrained.load_state_dict(copy_state_dict)
print("resnet18_pretrained")
print(resnet18_pretrained.state_dict()["fc.bias"])

        
