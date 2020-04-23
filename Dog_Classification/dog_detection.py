import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class DogDetection:
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        self.model = models.densenet161(pretrained=True)  # download Densenet-161 model

        # turn off training
        self.model.train(False)
        self.model.eval()

        # move model to gpu if available
        if self.use_cuda:
            self.model = self.model.cuda()

    def dog_detector(self, img_path):
        img = Image.open(img_path)

        # preprocess image
        img_to_tensor = transforms.Compose([transforms.Resize(224),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        img_tensor = img_to_tensor(img)
        img_tensor = img_tensor.view((1, *img_tensor.shape))

        # move image to gpu if available
        if self.use_cuda:
            img_tensor = img_tensor.cuda()

        prediction = self.model(img_tensor)

        return 151 <= torch.max(prediction, 1)[1].item() <= 268  # if class is between 151 and 268, it is a dog
