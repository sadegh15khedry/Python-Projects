'''''

if Mode == 'p':
    import pandas as pd
    import os
    import shutil
    import random

    test_number = [452,1287,332,86,262,23,25,62]
    val_number = [904,2575,664,173,524,47,50,125]
    class_number = ['0','1','2','3','4','5','6','7']

    
    for j in range(8):
        src = '/home/atlas/PycharmProjects/SimpleNet/ISIC_dataset/data/' + str(class_number[j])+'/'
        my_pics = os.listdir(path=src)
        #print(class_number[j])
        if j == 1:
            
            print("lets go!")
            dst = '/home/atlas/PycharmProjects/SimpleNet/ISIC_dataset/ISIC_binary_class/'
            for i in range(test_number[j]):
                pic_name = my_pics.pop(random.randrange(len(my_pics)))
                shutil.copyfile(src=src+pic_name, dst=dst+'test/1/' + str(pic_name))
            for i in range(val_number[j]):
                pic_name = my_pics.pop(random.randrange(len(my_pics)))
                shutil.copyfile(src=src+pic_name, dst=dst + 'val/1/' + str(pic_name))
            for i in my_pics:
                pic_name = i
                shutil.copyfile(src=src+pic_name, dst=dst + 'train/1/' + str(pic_name))
            
            print(class_number[j])
        else:
            print(class_number[j])
            print("lets go!")
            dst = '/home/atlas/PycharmProjects/SimpleNet/ISIC_dataset/ISIC_binary_class/'
            for i in range(test_number[j]):
                pic_name = my_pics.pop(random.randrange(len(my_pics)))
                shutil.copyfile(src=src + pic_name, dst=dst + 'test/0/' + str(pic_name))
            for i in range(val_number[j]):
                pic_name = my_pics.pop(random.randrange(len(my_pics)))
                shutil.copyfile(src=src + pic_name, dst=dst + 'val/0/' + str(pic_name))
            for i in my_pics:
                pic_name = i
                shutil.copyfile(src=src + pic_name, dst=dst + 'train/0/' + str(pic_name))


        
























    
    pics_src = '/home/atlas/PycharmProjects/SimpleNet/ISIC_dataset/ISIC_2019_Training_Input/'
    pics_des = '/home/atlas/PycharmProjects/SimpleNet/ISIC_dataset/data/'
    table = pd.read_csv('/home/atlas/PycharmProjects/SimpleNet/ISIC_dataset/ISIC_2019_Training_GroundTruth.csv')
    my_pics = os.listdir(path=pics_src)
    img_number = 0
    for i in range(table.shape[0]):
        filename = table.iloc[i]['image'] + '.jpg'
        if table.iloc[i]['MEL'] == 1:
            print("0")
            shutil.copyfile(src=pics_src+filename, dst=pics_des+'0/'+str(i) + "_" + str(0) + ".jpg")
        elif table.iloc[i]['NV'] == 1:
            print('1')
            shutil.copyfile(src=pics_src + filename, dst=pics_des + '1/' + str(i) + "_" + str(1) + ".jpg")
        elif table.iloc[i]['BCC'] == 1:
            print('2')
            shutil.copyfile(src=pics_src + filename, dst=pics_des + '2/' + str(i) + "_" + str(2) + ".jpg")
        elif table.iloc[i]['AK'] == 1:
            print('3')
            shutil.copyfile(src=pics_src + filename, dst=pics_des + '3/' + str(i) + "_" + str(3) + ".jpg")
        elif table.iloc[i]['BKL'] == 1:
            print('4')
            shutil.copyfile(src=pics_src + filename, dst=pics_des + '4/' + str(i) + "_" + str(4) + ".jpg")
        elif table.iloc[i]['DF'] == 1:
            print('5')
            shutil.copyfile(src=pics_src + filename, dst=pics_des + '5/' + str(i) + "_" + str(5) + ".jpg")
        elif table.iloc[i]['VASC'] == 1:
            print('6')
            shutil.copyfile(src=pics_src + filename, dst=pics_des + '6/' + str(i) + "_" + str(6) + ".jpg")
        elif table.iloc[i]['SCC'] == 1:
            print('7')
            shutil.copyfile(src=pics_src + filename, dst=pics_des + '7/' + str(i) + "_" + str(7) + ".jpg")
        elif table.iloc[i]['UNK'] == 1:
            print('8')
            shutil.copyfile(src=pics_src + filename, dst=pics_des + '8/' + str(i) + "_" + str(8) + ".jpg")
        























import shutil as sh
import os as os
import random
def binary_dataset_transfer():

    src_1 = '/home/atlas/PycharmProjects/DataSets/HAM10000/skin/train/bkl_1099'
    #test_number = [32,51,109,11,111,14]
    #val_number = [65,102,219,23,222,28]
    #class_name = ['akiec_327','bcc_514','bkl_1099','df_115','mel_1113','vasc_142']
    #class_number = [0,1,2,3,4,6]
    class_name = ['nv_6705']
    test_number = [670]
    val_number = [1341]
    class_number = [5]
    number = 3311
    for j in range(1):
        print(class_name[j])
        src_1 = '/home/atlas/PycharmProjects/DataSets/HAM10000/skin/train/'+class_name[j]
        my_pics = os.listdir(path=src_1)
        #test data class 0
        dst_1 = '/home/atlas/PycharmProjects/SimpleNet/skin_binary_class/test/1'
        for i in range(test_number[j]):
            pic_name = my_pics.pop(random.randrange(len(my_pics)))
            address = src_1+'/'+pic_name
            sh.copyfile(src=address, dst=dst_1 + "/"+str(number)+"_"+str(class_number[j])+".jpg")
            number = number+1

        #val data clsaa 0
        dst_1 = '/home/atlas/PycharmProjects/SimpleNet/skin_binary_class/val/1'
        for i in range(val_number[j]):
            pic_name = my_pics.pop(random.randrange(len(my_pics)))
            address = src_1+'/'+pic_name
            sh.copyfile(src=address, dst=dst_1 + "/" + str(number) + "_" + str(class_number[j]) + ".jpg")
            number = number+1
    #picking train data
        dst_1 = '/home/atlas/PycharmProjects/SimpleNet/skin_binary_class/train/1'
        for i in my_pics:
            address = src_1 + '/' + i
            sh.copyfile(src=address, dst=dst_1 + "/" + str(number) + "_" + str(class_number[j]) + ".jpg")
            number = number + 1
    print(number)




if Mode == "data_copy":
    binary_dataset_transfer()  
'''''