import torch
import torch.nn as nn
import torch.functional as F
import torchvision.models as models


class VGGNetwork(nn.Module):
    def __init__(self):
        super(VGGNetwork, self).__init__()
        self.encoder = models.vgg11(pretrained=True)

    def forward(self, x):
        return self.encoder(x)


if __name__ == "__main__":
    import numpy as np
    x = torch.tensor(np.random.random((110, 3, 224, 224)),dtype=torch.float32)
    net = VGGNetwork()
    res = net.forward(x)
    print(res)