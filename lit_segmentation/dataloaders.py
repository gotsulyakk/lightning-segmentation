import os
import cv2
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from albumentations.core.serialization import from_dict
import utils


class SegmentationDataset(Dataset):
    def __init__(
            self, 
            images_fps, 
            masks_fps, 
            transforms=None, 
    ):
        self.images_fps = images_fps
        self.masks_fps = masks_fps
        self.transforms = transforms
    
    def __getitem__(self, idx):        
        image = cv2.imread(self.images_fps[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[idx], cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.

        if self.transforms:
            sample = self.transforms(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
             
        return image, mask
        
    def __len__(self):
        return len(self.images_fps)


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        hparams,
    ):
        super().__init__()
        self.hparams = hparams

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            hparams["model"]["encoder_name"], 
            hparams["model"]["encoder_weights"]
        )
        
        self.train_ids = os.listdir(hparams["data"]["train_images_dir"])
        self.test_ids = os.listdir(hparams["data"]["test_images_dir"])

        self.train_images_fps = [
            os.path.join(hparams["data"]["train_images_dir"], image_id) for image_id in self.train_ids
        ]
        self.train_masks_fps = [
            os.path.join(hparams["data"]["train_masks_dir"], image_id) for image_id in self.train_ids
        ]

        self.test_images_fps = [
            os.path.join(hparams["data"]["test_images_dir"], image_id) for image_id in self.test_ids
        ]
        self.test_masks_fps = [
            os.path.join(hparams["data"]["test_masks_dir"], image_id) for image_id in self.test_ids
        ]    

        self.batch_size = hparams["train_parameters"]["batch_size"]
        self.train_augs = utils.get_validation_aug(self.preprocessing_fn)
        self.val_augs = utils.get_validation_aug(self.preprocessing_fn)
        self.test_augs = utils.get_validation_aug(self.preprocessing_fn)
        
    def setup(self, stage=None):       
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            self.train_images_fps, self.train_masks_fps, test_size=self.hparams["data"]["val_split"]
        )
        
        self.train_data = SegmentationDataset(train_imgs, train_masks, self.train_augs)
        self.val_data = SegmentationDataset(val_imgs, val_masks, self.val_augs)
        self.test_data = SegmentationDataset(self.test_images_fps, self.test_masks_fps, self.test_augs)
        
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"]
        )
        return train_loader
        
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"]
        )
        return val_loader
    
    def test_dataloder(self):
        test_loader = DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"]
        )
        return test_loader
