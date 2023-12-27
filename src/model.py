import torch
import pytorch_lightning as pl
import torchmetrics


class ClassificationMetric(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds, targets):
        correct = (preds == targets).sum()
        self.correct += correct
        self.total += len(targets)
    
    def compute(self):
        return self.correct / self.total
    
class ClassificationModel(pl.LightningModule):
    def __init__(self, model, n_classes=1000, pretrained=False, learning_rate=0.001):
        super().__init__()
        if model == "resnet18":
            from torchvision.models import resnet18 as resnet
            self.model = resnet(pretrained=pretrained)
            self.model.fc = torch.nn.Linear(512, n_classes)

        elif model == "resnet34":
            from torchvision.models import resnet34 as resnet
            self.model = resnet(pretrained=pretrained)
            self.model.fc = torch.nn.Linear(512, n_classes)

        elif model == "resnet50":
            from torchvision.models import resnet50 as resnet
            self.model = resnet(pretrained=pretrained)
            self.model.fc = torch.nn.Linear(2048, n_classes)
        else:
            raise NotImplementedError("model not supported")

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(num_classes=n_classes, top_k=1, task='multiclass')
        self.f1score = torchmetrics.F1Score(num_classes=n_classes, top_k=1, task='multiclass')
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx, 'train')
        
        accuracy = self.accuracy(y_hat.argmax(1), y)
        f1_score = self.f1score(y_hat, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1score': f1_score}, prog_bar=True,
                     on_step=False, on_epoch=True)
        return loss
        
    
    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx, 'valid')
        accuracy = self.accuracy(y_hat, y)
        f1_score = self.f1score(y_hat, y)
        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy, 'val_f1score': f1_score}, prog_bar=True,
                     on_step=False, on_epoch=True)
        return {'val_loss': loss, 'val_accuracy': accuracy, 'val_f1score': f1_score}

    
    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx, 'test')
        accuracy = self.accuracy(y_hat, y)
        f1_score = self.f1score(y_hat, y)
        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy, 'test_f1score': f1_score}, prog_bar=True,
                     on_step=False, on_epoch=True)
        return {'val_loss': loss, 'val_accuracy': accuracy, 'val_f1score': f1_score}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def _common_step(self, batch, batch_idx, split='train'):
        x, y = batch
        if self.current_epoch == 0:
            self.logger.log_image(f'{split}_grid', images=[x])
        
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y
        
    def predict(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        preds = torch.argmax(y_hat, dim=1)
        return y_hat, preds

    def on_train_epoch_end(self) -> None:
        print("\n")