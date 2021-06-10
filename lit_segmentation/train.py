from utils import object_from_dict
import yaml
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from dataloaders import SegmentationDataModule
import utils


class SegmentationModel(pl.LightningModule):   
    def __init__(
        self,
        hparams,
    ):
        super().__init__()

        self.hparams.update(hparams)
        
        self.model = object_from_dict(self.hparams["model"])       
        self.criterion = object_from_dict(self.hparams["criterion"])
        self.metric = object_from_dict(self.hparams["metric"])              
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = object_from_dict(
            self.hparams["scheduler"],
            optimizer=optimizer
            )

        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]

    
    def training_step(self, train_batch, batch_idx):     
        x, y = train_batch
        y = y.unsqueeze(1)
        
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)              
        score = self.metric(y_pred, y)

        logs = {'valid_loss': loss, 'valid_metrics': score}    
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, valid_batch, batch_idx):         
        x, y = valid_batch 
        y = y.unsqueeze(1)
        
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)      
        score = self.metric(y_pred, y)

        logs = {'valid_loss': loss, 'valid_metrics': score}    
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss


if __name__=='__main__':

    config_path = "configs/config.yaml"

    with open(config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    pl.utilities.seed.seed_everything(hparams["seed"])

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        hparams["model"]["encoder_name"], 
        hparams["model"]["encoder_weights"]
    )

    data = SegmentationDataModule(
        hparams=hparams,
        train_augs=utils.get_training_augmentation(preprocessing_fn),
        val_augs=utils.get_validation_augmentation(preprocessing_fn),
        test_augs=utils.get_validation_augmentation(preprocessing_fn)
    )

    model = SegmentationModel(
        hparams=hparams
    )

    trainer = object_from_dict(hparams["trainer"])
    trainer.fit(model, data)
