"""Data utility functions."""
import os
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

def load_mammography_data(img_name_file):
    path_to_images, _ = os.path.split(img_name_file)

    with open(img_name_file) as f:
        image_names = f.read().splitlines()

    image_names.remove(image_names[0])
    resize = transforms.Resize((240,240))
    gray_scale = transforms.Grayscale(num_output_channels=3)
    to_tensor = transforms.ToTensor()

    data = list()
    for i, img_name in enumerate(image_names):
        img = Image.open(os.path.join(path_to_images, img_name))
        img = resize(img)
        img = gray_scale(img)
        img = to_tensor(img)
        name, _ = os.path.splitext(img_name)
        if name.split('_')[2] == 'M':
            data.append((img, 1))
        elif name.split('_')[2] == 'B':
            data.append((img, 0))
        else:
            print(name, ': no label available')

    return data

