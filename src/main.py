import mlflow.pytorch
import pytorch_lightning as pl
from pprint import pprint

from data import MNISTDataModule
from model import LightningMNISTClassifier
from utils import print_auto_logged_info, fetch_logged_data
from mnist_classifier import MnistClassifier


model_path = '../data/model/'

def train_model(packaging=False):
    mnistdatamodule = MNISTDataModule()
    model = LightningMNISTClassifier(lr_rate=1e-3)

    trainer = pl.Trainer(max_epochs=30, default_root_dir=model_path, progress_bar_refresh_rate=20)
    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        trainer.fit(model, datamodule=mnistdatamodule)

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    for key, data in fetch_logged_data(run.info.run_id).items():
        print("\n--------- logged {} ---------".format(key))
        pprint(data)

    if packaging:
        bento_svc = MnistClassifier()
        bento_svc.pack('classifier', model)
        bento_svc.save()

if __name__ == '__main__':
    train_model(packaging=True)
