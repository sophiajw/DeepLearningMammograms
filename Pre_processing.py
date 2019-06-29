import os
import base64
import PIL
from PIL import Image
from glob import glob
import pandas as pd
import numpy as np

start_index = 0 #TODO: change accordingly
end_index = 322   #TODO: change accordingly
crop_length = 128 #default crop length
image_dim = 1024
counter = 1
info_file = pd.read_csv(r"./Info.txt", sep=' ')
for i in range(start_index, end_index):
    if(info_file.iloc[i, 2] != 'NORM' and not(np.isnan(info_file.iloc[i, 4]))):
        image_name = './Data/' + info_file.iloc[i,0] + '.pgm'
        image = Image.open(image_name)
        x = info_file.iloc[i,4]
        y = image_dim - info_file.iloc[i,5] 
        crop_length = info_file.iloc[i,6] + 50 #crop length according to the radius of tumor
        area = (x - crop_length, y - crop_length, x + crop_length, y + crop_length)
        image.crop(area).save('./processed-data/img' + str(counter).zfill(3) + '.jpg',"JPEG")
        counter = counter + 1