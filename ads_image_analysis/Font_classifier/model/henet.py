import torch 
import torch.nn as nn 
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

class HEBlock(nn.Module):
    def __init__(self, beta):
        super(HEBlock, self).__init__()
        self.beta = beta 
    
    def forward(self, X):
        """
        Input: 
            X's dimension is (batch_size, n_classes, height, width)
        
        Output:
            Output has the same dimension as the input.
        """

        dim = X.shape
        max_value, _ = torch.max(X.view(*dim[:2], -1), dim = -1)
        mask = X == max_value[:, :, None, None]
        X[mask] *= self.beta
        return X


class HENet(nn.Module):
    """
    https://arxiv.org/pdf/2110.10872v1
    ResNet18 + HE Block.
    """

    def __init__(self, n_classes, block_expansion=1, theta=0.5):
        super(HENet, self).__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = 512 

    # Architecture: 
        # Backbone: ResNet18
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # HE Block
        self.reduce_channel = nn.Conv2d(in_features, int(n_classes*block_expansion), 1, 1)
        self.heblock = HEBlock(theta)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        # Classifier 
        self.fc = nn.Linear(int(n_classes*block_expansion), n_classes)



        self.__set_params()

    def __set_params(self):
        self.backbone_params = list(self.backbone.parameters())
        self.head_params = [p for n, p in self.named_parameters() if 'backbone' not in n]


    def forward(self, X):
        """
        Input:
            X dimension is (batch_size, n_classes, height, width)
        
        Output: 
            Output dimension is (batch_size, n_classes)
        """
        feature_map = self.reduce_channel(self.backbone(X))
        if self.training:
            feature_map = self.heblock(feature_map)
        out = self.pool(feature_map)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    

# Debug 
if __name__ == "__main__":
    n_classes = 100
    test_data = torch.rand(64, 3, 224, 224)
    model = HENet(n_classes=n_classes)
    output = model.forward(test_data, training=True)
    print(output)
    print(output.shape)
