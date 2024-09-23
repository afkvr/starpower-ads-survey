import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from .similarity_module import ContrastiveEmbedding, LightContrastiveEmbedding
from torchmetrics import Accuracy, F1Score, Precision, Recall
from pytorch_lightning.utilities.grads import grad_norm
from pytorch_metric_learning import miners, losses, distances, reducers
from pytorch_metric_learning.regularizers import LpRegularizer



def L1(X, Y): 
    return torch.linalg.norm(X - Y, dim=1, ord=1)

def L2(X, Y): 
    return torch.linalg.norm(X - Y, dim=1, ord=2)



class EncoderWrapper(L.LightningModule):
    def __init__(self, model, embedding_size: int, learning_rate: float=0.01, margin: float=0.5):
        super().__init__()

        self.learning_rate = learning_rate 
        self.embedding_size = embedding_size 
        self.margin = margin

        self.model = model 

        self.example_input_array = torch.randn(1, 3, 224, 224)

    def forward(self, X): 
        return self.model(X)
    
    def training_step(self, batch, batch_idx): 
        self.model.train()
        X, Labels = batch

        Embeddings = F.normalize(self(X), p=2, dim=1)

        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)
        
        miner = miners.TripletMarginMiner(margin=self.margin, type_of_triplets="all")

        loss_fn = losses.TripletMarginLoss(margin=self.margin, distance= distance, reducer=reducer)
        
        hard_pairs = miner(Embeddings, Labels)
        loss = loss_fn(Embeddings, Labels, hard_pairs)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        X, Labels = batch

        Embeddings = F.normalize(self(X), p=2, dim=1)

        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)
        
        miner = miners.TripletMarginMiner(margin=self.margin, type_of_triplets="all")

        loss_fn = losses.TripletMarginLoss(margin=self.margin, distance= distance, reducer=reducer)
        
        hard_pairs = miner(Embeddings, Labels)
        loss = loss_fn(Embeddings, Labels, hard_pairs)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return {"val_loss": loss}



    def on_before_optimizer_step(self, optimizer):
        # Compute the L2 norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)


    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            params, lr=self.learning_rate, weight_decay=0.001
        )
        scheduler = StepLR(
            optimizer, step_size=5, gamma=0.8
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }




# Debug 
if __name__ == "__main__":
    embedding_size = 256
    # model = ContrastiveEmbeddingWrapper(model=ContrastiveEmbedding(embedding_size= embedding_size),
                        #  embedding_size= embedding_size,
                        #  margin=1.5,
                        #  learning_rate=0.001
                        #  )
    
    # print(model)