import base64
import PIL
from PIL import Image
import pandas as pd
import numpy as np
import random

def split_data (train_pro=0.6,val_pro=0.2,test_pro=0.2):
    #At first we only want to use the normal data
    info_file = pd.read_csv(r"./data/Pre_processed_data.txt", sep='_')
    #For adding the augemented data, change the line above in the line below
    #info_file = pd.read_csv(r"./data/Data_processing_augmentation.txt", sep='_')
    
    counter_B=0
    counter_M=0
    
    #Count how much begin and how much malignant cases we have
    for i in range(len(info_file)):
        if (info_file.iloc[i,1]=='B'):
            counter_B = counter_B +1
        elif(info_file.iloc[i,1]=='M'):
            counter_M = counter_M +1
    
    #Compute how big each set should be   
    train_B = round(counter_B*train_pro)
    train_M = round(counter_M*train_pro)
    val_B = round(counter_B*val_pro)
    val_M = round(counter_M*val_pro)
    test_B = round(counter_B*test_pro)
    test_M = round(counter_M*test_pro)

    sum_B= test_B+val_B+train_B
    sum_M= test_M+val_M+train_M

    #Correcting the rounding mistake
    if(sum_B!= counter_B):
        train_B = train_B-abs(sum_B-counter_B)
    if(sum_M != counter_B):
        train_M = train_M-abs(sum_M-counter_M)

    #Begin list w
    Blist=list()
    for i in range (len(info_file)):
        if (info_file.iloc[i,1]=='B'):
            Blist.append('img'+ '_' + str(info_file.iloc[i,0]) +'_' +info_file.iloc[i,1]+ '.jpg')
    
    #Malignant list
    Mlist=list()
    for i in range (len(info_file)):
        if (info_file.iloc[i,1]=='M'):
            Mlist.append('img'+ '_' + str(info_file.iloc[i,0]) +'_' +info_file.iloc[i,1] + '.jpg')

    #train list
    train_list = list()
    train_list = Blist[0:train_B]
    train_list[train_B +1 : train_M] = Mlist[0:train_M]

    #Shuffling the training list
    random.seed(1223)
    random.shuffle(train_list)


    #validation list
    val_list = list()
    val_list = Blist[train_B : train_B + val_B]
    val_list[val_B+1 : val_M] = Mlist[train_M : train_M+ val_M]

    #Shuffeling the validation list
    random.seed(123)
    random.shuffle(val_list)

    #test list 
    test_list = list()
    test_list = Blist[train_B + val_B : train_B + val_B + test_B]
    test_list[test_B+1 : test_M] = Mlist[train_M+ val_M:train_M+ val_M+test_M]

    #Shuffeling the test list
    random.seed(12)
    random.shuffle(test_list)


    #Saving train, validation and test list as an txt file
    with open('./data/train.txt', 'w') as f:
        for item in train_list:
            f.write("%s\n" % item)
        
    with open('./data/val.txt', 'w') as f:
        for item in val_list:
            f.write("%s\n" % item)
        
    with open('./data/test.txt', 'w') as f:
        for item in test_list:
            f.write("%s\n" % item)





