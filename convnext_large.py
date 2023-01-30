import torch
SEED = 1203
BATCH_SIZE = 16
TEST_SIZE = 0.1
NUM_GPUS = torch.cuda.device_count()
EPOCHS = 40
ADD_SIZE = 64
DROP_RATE = 0.2
LEARNING_RATE = 0.00009 # 

                                    # sample per sec  / num params(m)
# model_name = 'coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k' # 1025.45	41.72
# model_name = 'maxvit_base_tf_224.in1k'   # 358.25	119.47	
# model_name = 'coatnet_2_rw_224.sw_in12k_ft_in1k' # 631.88	73.87	
# model_name = 'maxvit_base_tf_384.in21k_ft_in1k'
# model_name = 'maxvit_nano_rw_256.sw_in1k'
# model_name = "swinv2_large_window12to24_192to384_22kft1k"
# model_name = 'maxvit_large_tf_384.in21k_ft_in1k'
model_name = 'convnext_large.fb_in22k_ft_in1k_384' # 104.71	119.65

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", dest="seed", action="store", default=SEED)
args = parser.parse_args()
SEED = int(args.seed)

folder_name = f"./checkpoints/{model_name}_{SEED}"

import os
import warnings
from datetime import datetime
from typing import Callable

import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torchvision
from loguru import logger
from PIL import Image
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from rich.traceback import install
from sklearn.model_selection import train_test_split
from timm import create_model
from timm.optim import create_optimizer_v2
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MultilabelAccuracy
from torchvision import transforms as T
install(show_locals=True, suppress=["torch", "timm", "pytorch_lightning"])
warnings.filterwarnings("ignore", category=UserWarning)
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED) # Seed 고정

def train_augment(size, mean, std):
    return T.Compose(
        [
            T.RandomChoice([
                T.RandomHorizontalFlip(p=0.8),
                T.RandomApply(
                    [
                        T.RandomAffine(
                            degrees=10,
                            translate=(0.1, 0.1),
                            scale=(0.9, 1.1),
                            shear=10,
                            fill=255,
                            interpolation=T.InterpolationMode.BICUBIC,
                        )
                    ],
                    p=0.8,
                ),                
            ]),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.CenterCrop(size=size),
            T.AugMix(),  # torchvision>=0.13.0 필요
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def train_transform(size, mean, std):
    return T.Compose(
        [
            T.Resize(size=size+ADD_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size=size),
            T.AugMix(),  # torchvision>=0.13.0 필요
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def val_transform(size, mean, std):
    return T.Compose(
        [
            T.Resize(size=size+ADD_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size=size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    
class BlockDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Callable):
        self.df = df
        self.transform = transform
        self.is_predict = self.df.shape[1] < 3

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row["img_path"]

        if not self.is_predict:
            labels = row["A":"J"].to_numpy(dtype=np.float32)
        else:
            labels = np.zeros(10, dtype=np.float32)
        img = Image.open(image_path)
        img = self.transform(img)
        return img, torch.tensor(labels, dtype=torch.float32)
    
    
class BlockDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ):
        super().__init__()
        self.train_df = pd.read_csv("./data/train.csv")
        self.train_df['img_path'] = self.train_df['img_path'].str.replace('./train', '/home/gyuseonglee/workspace/play/data/train_bg').str.replace('TRAIN', 'BG_TRAIN').values
        self.test_df = pd.read_csv("./data/test.csv")
        self.test_df['img_path'] = self.test_df['img_path'].str.replace('./test', '/home/gyuseonglee/workspace/play/data/test_bg').str.replace('TEST', 'BG_TEST').values
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform(size, mean, std)
        self.train_augment = train_augment(size, mean, std)
        self.val_transform = val_transform(size, mean, std)
        self.test_transform = val_transform(size, mean, std)

    def setup(self, stage=None):
        if stage == "fit":
            df = pd.read_csv("./data/train.csv")
            df['img_path'] = df['img_path'].str.replace('./train', '/home/gyuseonglee/workspace/play/data/train_bg').str.replace('TRAIN', 'BG_TRAIN').values

            train_df, val_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED) 
            self.train_dataset = torch.utils.data.ConcatDataset([
                BlockDataset(train_df, self.train_transform), BlockDataset(train_df, self.train_augment)
            ])
            self.val_dataset = BlockDataset(val_df, self.val_transform)

        if stage == "predict":
            df = pd.read_csv("./data/test.csv")
            df['img_path'] = df['img_path'].str.replace('./test', '/home/gyuseonglee/workspace/play/data/test_bg').str.replace('TEST', 'BG_TEST').values
            
            self.test_dataset = BlockDataset(df, self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )


class MyModule(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_accuracy = MultilabelAccuracy(10)
        self.val_accuracy = MultilabelAccuracy(10)

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(
        #     self.model.parameters(), lr=self.hparams.lr, weight_decay=0.01
        # )
        optimizer = create_optimizer_v2(
            self.model, "madgradw", lr=self.hparams.lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
        }

        return [optimizer], [scheduler_config]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_accuracy(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log(
            "train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_accuracy(logits, y)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        pred = torch.sigmoid(logits)
        return pred


def main():
    logger.info("training start")
    logger.info(f"pytorch version: {torch.__version__}")
    logger.info(f"torchvision version: {torchvision.__version__}")

    logger.info("load model")

    model = create_model(model_name, pretrained=True, num_classes=10, drop_rate=DROP_RATE)
    if os.path.isfile(folder_name):
        L = os.listdir(folder_name)
        max_epoch = 0
        for f in L:
            if ('epoch' in f):
                cur_epoch = int(f.split("epoch=")[1].split('-step')[0])
                if cur_epoch >max_epoch:
                    max_epoch = cur_epoch
                    load_target = f
        if max_epoch != 0:
            model = MyModule.load_from_checkpoint(f)
            print(f"-- checkpoint loaded, starting epoch: {max_epoch}")

    
    config = model.default_cfg
    size = config["input_size"][1]
    mean = config["mean"]
    std = config["std"]

    logger.debug(f"{size = }, {mean = }, {std = }")

    logger.info("load data")
    datamodule = BlockDataModule(
        batch_size=BATCH_SIZE, num_workers=12, size=size, mean=mean, std=std
    )

    logger.info("create module")
    module = MyModule(model, lr=LEARNING_RATE)


    checkpoints = ModelCheckpoint(dirpath=folder_name, monitor="val_acc", mode="max")
    callbacks = [checkpoints, RichProgressBar(), LearningRateMonitor()]

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer = pl.Trainer(
        gpus=NUM_GPUS,
        accelerator="gpu", strategy="ddp",
        logger=WandbLogger(name=f"{model_name}_{now}_{SEED}", project="4dblock"),
        callbacks=callbacks,
        # callbacks=None,
        max_epochs=EPOCHS,
        precision=16,
        # fast_dev_run=True,
    )

    logger.info("start training")
    trainer.fit(module, datamodule=datamodule)
    logger.info("training end")
    
    
    best_model_path = trainer.checkpoint_callback.best_model_path

    torch.distributed.destroy_process_group()
    if best_model_path and os.path.isfile(best_model_path):
        if trainer.global_rank == 0:
            trainer = pl.Trainer(
                gpus=1,
                accelerator="gpu",# strategy="bagua",
                logger=WandbLogger(name=f"{model_name}_{now}", project="4dblock"),
                callbacks=callbacks,
                # callbacks=None,
                max_epochs=EPOCHS,
                precision=16,
                # fast_dev_run=True,
            )                
            logger.info("load best model")
            best_model = MyModule.load_from_checkpoint(best_model_path)
            logger.info("start predict")
            
            
            raw_pred = trainer.predict(best_model, datamodule=datamodule)
            prob = torch.cat(raw_pred).cpu().numpy()

            pred = np.where(prob > 0.5, 1, 0)

            submission = pd.read_csv("/home/gyuseonglee/workspace/play/data/sample_submission.csv")
            submission.iloc[:, 1:] = pred

            file_name = f"{folder_name}/{model_name}_submission_{now}.csv"
            logger.info(f"save submission file: {file_name}")
            submission.to_csv(file_name, index=False)


if __name__ == "__main__":
    main()


