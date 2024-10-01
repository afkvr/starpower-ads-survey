import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from .henet import HENet 
from torchmetrics import Accuracy, F1Score, Precision, Recall
from pytorch_lightning.utilities.grads import grad_norm




class HENetWrapper(L.LightningModule):
    def __init__(self, model: HENet, num_classes: int, learning_rate: float = 0.01):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average = "macro")
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes, average = "macro")
        self.precision = Precision(task="multiclass", num_classes=num_classes, average = "macro")
        self.recall = Recall(task="multiclass", num_classes=num_classes, average = "macro")

        self.eval_loss: list[torch.Tensor] = []
        self.eval_accuracy: list[torch.Tensor] = []

        self.test_accuracy: list[torch.Tensor] = []

        self.example_input_array = torch.zeros(5, 3, 224, 224)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        train_acc = self.accuracy(y_hat.argmax(1), y)
        train_f1 = self.f1(y_hat.argmax(1), y)
        train_pre = self.precision(y_hat.argmax(1), y)
        train_rec = self.recall(y_hat.argmax(1), y)

        self.log("train_acc", train_acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_f1", train_f1, prog_bar=True)
        self.log("train_pre", train_pre, prog_bar=True)
        self.log("train_rec", train_rec, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1, pre, rec = self._shared_eval(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.eval_loss.append(loss)
        self.eval_accuracy.append(acc)
        return {"val_loss": loss, "val_accuracy": acc, "val_f1": f1, "val_precision": pre, "val_recall": rec}

    def test_step(self, batch, batch_idx):
        loss, acc, f1, pre, rec = self._shared_eval(batch)
        self.log_dict({"test_loss": loss, "test_acc": acc, "test_f1": f1, "test_precision": pre, "test_recall": rec})

    def _shared_eval(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat.argmax(1), y)
        f1 = self.f1(y_hat.argmax(1), y)
        pre = self.precision(y_hat.argmax(1), y)
        rec = self.recall(y_hat.argmax(1), y)
        return loss, acc, f1, pre, rec

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.SGD(
            params, lr=self.learning_rate, momentum=0.9, weight_decay=0.0001
        )
        scheduler = ReduceLROnPlateau(
            optimizer, "min", patience=3
        )
        scheduler1 = StepLR(
            optimizer, step_size=5, gamma=0.8
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler1,
                "monitor": "val_loss",
            },
        }


# Debug 
if __name__ == "__main__":
    n_classes = 100
    model = HENetWrapper(model=HENet(n_classes=n_classes),
                         num_classes=n_classes,
                         learning_rate=0.0001
                         )
    
