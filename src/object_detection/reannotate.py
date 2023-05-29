import os


import cv2
import random
import shutil

current_dir = os.getcwd()
train_dir = os.path.join(current_dir, 'yolov5', 'custom_dataset', 'labels', 'train')
val_dir = os.path.join(current_dir, 'yolov5', 'custom_dataset', 'labels', 'val')
#test_dir = os.path.join(current_dir, 'yolov5', 'custom_dataset','labels', 'test')

labels_dir = os.path.join(current_dir, 'yolov5', 'custom_dataset', 'labels')

dest_train_path = os.path.join(current_dir, 'yolov5', 'custom_dataset', 'images' , 'train')
dest_val_path = os.path.join(current_dir, 'yolov5', 'custom_dataset', 'images' , 'val')
#dest_test_path = os.path.join(current_dir, 'yolov5', 'custom_dataset', 'images' , 'test')
dest_labels_train_path = os.path.join(current_dir, 'yolov5', 'custom_dataset', 'labels' , 'train')
dest_labels_val_path = os.path.join(current_dir, 'yolov5', 'custom_dataset', 'labels' , 'val')
#dest_labels_test_path = os.path.join(current_dir, 'yolov5', 'custom_dataset', 'labels' , 'test')

#print(os.listdir(train_dir))
label_files = os.listdir(train_dir)

for label_file in label_files:
    data = []
    with open(os.path.join(train_dir, label_file), 'r') as wf:
        #print(os.path.join(train_dir, label_file))
        for line in wf:
            line_split = line.split(" ")
            line_new = '0 ' + line_split[1] + ' ' + line_split[2] + ' ' + line_split[3] + ' ' + line_split[4]
            data.append(line_new)
            
    with open(os.path.join(train_dir, label_file), 'w') as wf:
        wf.writelines(data)  
        
        
label_files = os.listdir(val_dir)

for label_file in label_files:
    data = []
    with open(os.path.join(val_dir, label_file), 'r') as wf:
        print(os.path.join(val_dir, label_file))
        for line in wf:
            line_split = line.split(" ")
            print(line_split)
            if len(line_split) <=3:
                continue
            line_new = '0 ' + line_split[1] + ' ' + line_split[2] + ' ' + line_split[3] + ' ' + line_split[4]
            data.append(line_new)
            
    with open(os.path.join(val_dir, label_file), 'w') as wf:
        wf.writelines(data)
        
#label_files = os.listdir(test_dir)

#for label_file in label_files:
#    data = []
#    with open(os.path.join(test_dir, label_file), 'r') as wf:
#        print(os.path.join(test_dir, label_file))
#        for line in wf:
#            line_split = line.split(" ")
#            line_new = '0 ' + line_split[1] + ' ' + line_split[2] + ' ' + line_split[3] + ' ' + line_split[4]
#            data.append(line_new)
            
#    with open(os.path.join(test_dir, label_file), 'w') as wf:
#        wf.writelines(data)  
        
 
