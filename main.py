import os

import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger

from src.model import ClassificationModel
from src.dataset import ImageDataset
from src.callbacks import checkpoint_callback, early_stop

def main():
    from yacs.config import CfgNode as CN
    import yaml
    config = CN(yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader))

    model_name = config.model.model_name
    n_classes = config.model.num_classes
    img_size = config.model.image_size
    pertrained = config.model.pretrained

    batch_size = config.training.batch_size
    num_workers = config.training.num_workers
    epochs = config.training.epochs
    learning_rate = config.training.learning_rate

    data_root = config.data.data_folder
    split_val_test = config.data.split_val_test
    log_dir = config.data.logs_dir

    project_name = config.project_name
    gpus_ids = config.gpus_ids

    os.makedirs(log_dir, exist_ok=True)
    
    wandb_logger = WandbLogger(project=project_name, save_dir=log_dir, log_model=True)

    call_backs = [checkpoint_callback(), early_stop()]

    wandb_logger.experiment.config.update({"batch_size": batch_size,
                                           "learning_rate": learning_rate,
                                           "epochs": epochs,
                                           "model_name": model_name})

    assert os.path.exists(data_root), "data folder not found"

    model = ClassificationModel(model_name, n_classes=n_classes, pretrained=pertrained, learning_rate=learning_rate)

    data_module = ImageDataset(data_root, img_size=img_size, batchsize=batch_size, num_workers=num_workers,
                               split_train_val_test=split_val_test)

    trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', devices=gpus_ids, enable_checkpointing=True,
                         default_root_dir=log_dir, logger=wandb_logger, callbacks=call_backs)
    
    trainer.fit(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()