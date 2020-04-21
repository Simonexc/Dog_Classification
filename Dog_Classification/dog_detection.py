import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class DogDetection:
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        self.model = models.densenet161(pretrained=True)
        self.model.train(False)
        self.model.eval()

        if self.use_cuda:
            self.model = self.model.cuda()

    def model_prediction(self, img_path):
        img = Image.open(img_path)
        img_to_tensor = transforms.Compose([transforms.Resize(224),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        img_tensor = img_to_tensor(img)
        img_tensor = img_tensor.view((1, *img_tensor.shape))
        if self.use_cuda:
            img_tensor = img_tensor.cuda()

        prediction = self.model(img_tensor)

        return torch.max(prediction, 1)[1].item()  # predicted class index

    def dog_detector(self, img_path):
        predicted_index = self.model_prediction(img_path)

        return 151 <= predicted_index <= 268
