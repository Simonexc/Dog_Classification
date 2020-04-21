import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from .class_names import class_names


class DogClassification:
    def __init__(self, use_cuda, app_root):
        self.use_cuda = use_cuda
        self.app_root = app_root
        self.class_names = class_names

        self.model = models.densenet161(pretrained=True)
        for param in self.model.features.parameters():
            param.requires_grad = False

        n_inputs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_inputs, 133)

        trained_model_path = os.path.join(self.app_root, 'static/models/model_transfer.pt')
        self.model.load_state_dict(torch.load(trained_model_path))

        self.model.train(False)
        self.model.eval()

        if self.use_cuda:
            self.model = self.model.cuda()

    def predict_breed(self, img_path):
        img = Image.open(img_path)
        img_to_tensor = transforms.Compose([transforms.Resize(224),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        img_tensor = img_to_tensor(img)
        img_tensor = img_tensor.view((1, *img_tensor.shape))
        if self.use_cuda:
            img_tensor = img_tensor.cuda()

        output = self.model(img_tensor)
        output = torch.exp(output)
        prediction = torch.max(output, 1)[1].item()

        return self.class_names[prediction]
