import os
import base64
import PIL
from PIL import Image
from glob import glob
import pandas as pd
import numpy as np

start_index = 0 #TODO: change accordingly
crop_length = 128 #default crop length
image_dim = 1024
counter = 1
info_file = pd.read_csv(r"./data/Info.txt", sep=' ')
end_index = info_file.shape[0]   #TODO: change accordingly
prepro = list()
prepro.append('Number_Case')

for i in range(start_index, end_index):
    if(info_file.iloc[i, 2] != 'NORM' and not(np.isnan(info_file.iloc[i, 4]))):
        image_name = './Data_original/' + info_file.iloc[i,0] + '.pgm'
        image = Image.open(image_name)
        x = info_file.iloc[i,4]
        y = image_dim - info_file.iloc[i,5] 
        crop_length = info_file.iloc[i,6] + 50 #crop length according to the radius of tumor
        area = (x - crop_length, y - crop_length, x + crop_length, y + crop_length)
        image.crop(area).save('./data/img'+'_' + str(counter) +'_' +info_file.iloc[i,3] + '.jpg',"JPEG")
        prepro.append('img' + '_' + str(counter) + '_' +info_file.iloc[i,3])
        counter = counter + 1
with open('./data/Pre_processed_data.txt', 'w') as f:
    for item in prepro:
        f.write("%s\n" % item)
        
        

info_file_pro = pd.read_csv(r"./data/Pre_processed_data.txt",sep = '_')
end_index = info_file_pro.shape[0]

start_index = 0
saving_path = './data'

aug = list()
aug.append('Number_Case')

for i in range(start_index, end_index):
    image_name = './data/img' + '_' + str(info_file_pro.iloc[i,0]) + '_' + info_file_pro.iloc[i,1] 
    image = Image.open(image_name + '.jpg')
    
    #Rotation 90 degrees
    image90 = image.rotate(90)
    #Rotation 180 degrees
    image180 = image.rotate(180)
    #Rotation 270 degrees
    image270 = image.rotate(270)

    #X for 90 degrees rotation
    image.rotate(90).save('./data/img' + '_' + str(info_file_pro.iloc[i,0]) + 'X' + '_' 
                          +  info_file_pro.iloc[i,1]+'.jpg',"JPEG")
    aug.append('img' + '_' + str(info_file_pro.iloc[i,0]) + 'X_' + info_file_pro.iloc[i,1])
    prepro.append('img' + '_' + str(info_file_pro.iloc[i,0]) + 'X_' + info_file_pro.iloc[i,1])
    
    #Y for 180 degrees rotation
    image.rotate(180).save('./data/img' + '_' + str(info_file_pro.iloc[i,0]) + 'Y' + '_' 
                           + info_file_pro.iloc[i,1]+'.jpg',"JPEG")
    aug.append('img' + '_' + str(info_file_pro.iloc[i,0]) + 'Y_' + info_file_pro.iloc[i,1])
    prepro.append('img' + '_' + str(info_file_pro.iloc[i,0]) + 'Y_' + info_file_pro.iloc[i,1])
    
    #Z for 270 degrees rotation
    image.rotate(270).save('./data/img' + '_' + str(info_file_pro.iloc[i,0]) + 'Z' + '_' 
                           + info_file_pro.iloc[i,1]+'.jpg',"JPEG")
    aug.append('img' + '_' + str(info_file_pro.iloc[i,0]) + 'Z_' + info_file_pro.iloc[i,1])
    prepro.append('img' + '_' + str(info_file_pro.iloc[i,0]) + 'Z_' + info_file_pro.iloc[i,1])
    
    
with open('./data/Augmentation.txt', 'w') as f:
    for item in aug:
        f.write("%s\n" % item)

with open('./data/Data_processing_augmentation.txt', 'w') as f:
    for item in prepro:
        f.write("%s\n" % item)
        
        