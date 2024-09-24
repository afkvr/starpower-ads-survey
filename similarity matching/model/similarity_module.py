import torch 
import torch.nn as nn 
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights


class ProjectionHead(nn.Module): 
    def __init__(self, input_size, output_size): 
        super(ProjectionHead, self).__init__()
        self.input_size  = input_size 
        self.output_size = output_size

    # Architecture 
        self.Head = nn.Sequential( 
            nn.Linear(input_size, 1024), 
            nn.BatchNorm1d(1024),
            nn.GELU(),

            nn.Linear(1024, 512), 
            nn.BatchNorm1d(512),
            nn.GELU(),
            
            nn.Linear(512, self.output_size)
        )

    def forward(self, X): 
        return self.Head(X)
        
class ContrastiveEmbedding(nn.Module): 
    def __init__(self, embedding_size=128, req_grad=False): 
        super(ContrastiveEmbedding, self).__init__()
        self.req_grad = req_grad
        
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        for params in self.backbone.parameters():
            params.requires_grad = self.req_grad

        self.encoder_features = resnet.fc.in_features
        self.embedding_size = embedding_size
        
        self.ProjectionHead = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.encoder_features, self.embedding_size)
        )

    def forward(self, X1):
        embedding = torch.flatten(self.backbone(X1), start_dim=1)
        embedding = self.ProjectionHead(embedding1)

        return embedding

class LightContrastiveEmbedding(nn.Module): 
    def __init__(self, embedding_size=128, req_grad=False): 
        super(LightContrastiveEmbedding, self).__init__()
        self.req_grad = req_grad

        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.encoder_features = resnet.fc.in_features
        self.embedding_size = embedding_size
        
        for params in self.backbone.parameters():
            params.requires_grad = self.req_grad


        self.ProjectionHead = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.encoder_features, self.embedding_size)
        )

    def forward(self, X1):
        embedding = torch.flatten(self.backbone(X1), start_dim=1)
        embedding = self.ProjectionHead(embedding)

        return embedding


# Debug
if __name__=="__main__": 
    # model = LightContrastiveEmbedding(embedding_size=128)
    # model.train()

    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # int(len(list(resnet.children()))/2)]

    test_image = torch.rand(5, 3, 1024, 1024)
    embedding = backbone(test_image)


    print(embedding.shape)
