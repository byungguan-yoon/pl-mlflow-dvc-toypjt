from typing import BinaryIO, List
from PIL import Image
from torchvision import transforms
import torch
import bentoml
from bentoml.frameworks.pytorch import PytorchLightningModelArtifact
from bentoml.adapters import FileInput, JsonOutput

MNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([PytorchLightningModelArtifact('classifier')])
class MnistClassifier(bentoml.BentoService):
    @bentoml.utils.cached_property
    def transform(self):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    @bentoml.api(input=FileInput(), output=JsonOutput(), batch=True)
    def predict(self, file_streams: List[BinaryIO]) -> List[str]:
        img_tensors = []
        for fs in file_streams:
            img = Image.open(fs).convert(mode="L").resize((28, 28))
            img_tensors.append(self.transform(img))
        outputs = self.artifacts.classifier(torch.stack(img_tensors))
        _, output_classes = outputs.max(dim=1)
        return [MNIST_CLASSES[output_class] for output_class in output_classes]