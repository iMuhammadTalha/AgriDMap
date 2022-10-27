from django.http import HttpResponse
from django.shortcuts import render
import joblib
import torch
from django.core.files.storage import default_storage
from PIL import Image
from torchvision import datasets, models, transforms
from torch.autograd import Variable

test_image1 = "image12.png"

def home(request):
    if request.method == "POST":
        model = torch.load('resnet_test_model_new.pt',map_location=torch.device('cpu'))
        model.eval()

        file = request.FILES["pic"]
        file_name = default_storage.save(file.name, file)
        file_url = default_storage.path(file_name)

        image = image_loader(file_url)

        # image = image_loader(test_image1) # Test File
        predict = model(image)
        # print(predict.shape)
        # print(predict)
        _,index = torch.max(predict,1)
        pred = torch.max(predict.data,1)
        values,indices = pred
        print(indices)

        if indices == 0:
            result = "HEALTHY"
        elif indices == 1:
            result = "RESISTANT"
        elif indices == 2:
            result = "SUSCEPTIBLE"

        print(result)

        return render(request, "home.html", {'ResultClass':result, 'FileURL': file_url})
    else:
        return render(request, "home.html")

def result(request):

    model = torch.load('resnet_test_model_new.pt',map_location=torch.device('cpu'))
    model.eval()

    file = request.FILES["pic"]
    file_name = default_storage.save(file.name, file)
    file_url = default_storage.path(file_name)

    image = image_loader(file_url)

    # image = image_loader(test_image1) # Test File
    predict = model(image)
    # print(predict.shape)
    # print(predict)
    _,index = torch.max(predict,1)
    pred = torch.max(predict.data,1)
    values,indices = pred
    print(indices)

    if indices == 0:
        result = "HEALTHY"
    elif indices == 1:
        result = "RESISTANT"
    elif indices == 2:
        result = "SUSCEPTIBLE"

    print(result)

    return render(request, "home.html", {'ResultClass':result, 'FileURL': file_url})

Test_data_transforms = transforms.Compose([ transforms.Resize(size=(224,224)),
     transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])]) #Normalizing the data to the data that the ResNet18 was trained on])


def image_loader(image_name):
    """load image"""
    image = Image.open(image_name).convert('RGB')
    image = Test_data_transforms(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  
    return image