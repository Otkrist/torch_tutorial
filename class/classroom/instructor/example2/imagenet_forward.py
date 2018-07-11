from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy

def imshow(img):
    npimg = numpy.asarray(img)
    print("Displaying Image")
    plt.imshow(npimg)
    plt.show()

# Load model
squeeze = models.squeezenet1_1(pretrained=True)
#squeeze = models.resnet18(pretrained=True)

# Data transformations.
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Scale(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

# Load image.
img_pil = Image.open(open("input.jpg","r"))
imshow(img_pil)
img_tensor = preprocess(img_pil)
img_tensor.unsqueeze_(0)

# Send image through network.
img_variable = Variable(img_tensor)
fc_out = squeeze(img_variable)

from labels import labels
print(labels[str(fc_out.data.numpy().argmax())])

