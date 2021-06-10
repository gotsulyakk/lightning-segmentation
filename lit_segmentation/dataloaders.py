import os
import cv2
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


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
        train_augs=None,
        val_augs=None,
        test_augs=None
    ):
        super().__init__()

        self.hparams = hparams
        
        self.ids = os.listdir(hparams["data"]["images_dir"])
        self.images_fps = [
            os.path.join(hparams["data"]["images_dir"], image_id) for image_id in self.ids
            ]
        self.masks_fps = [
            os.path.join(hparams["data"]["masks_dir"], image_id) for image_id in self.ids
            ]
        self.batch_size = hparams["train_parameters"]["batch_size"]
        self.train_augs = train_augs
        self.val_augs = val_augs
        self.test_augs = test_augs
        
    def setup(self, stage=None):
        
        train_imgs, test_imgs, train_masks, test_masks = train_test_split(
            self.images_fps, self.masks_fps, test_size=0.1
            )
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            train_imgs, train_masks, test_size=self.hparams["data"]["val_split"]
            )
        
        self.train_data = SegmentationDataset(train_imgs, train_masks, self.train_augs)
        self.val_data = SegmentationDataset(val_imgs, val_masks, self.val_augs)
        self.test_data = SegmentationDataset(test_imgs, test_masks, self.test_augs)
        
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"])

        return train_loader
        
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"])

        return val_loader
    
    def test_dataloder(self):
        test_loader = DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"])

        return test_loader
