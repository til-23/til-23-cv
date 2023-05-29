import os

print(os.listdir())

current_dir = os.getcwd()
train_dir = os.path.join(current_dir, 'yolov5', 'custom_dataset', 'images', 'train')

img_names = os.listdir(train_dir)

with open('train.txt', 'w') as f:
    for img_name in img_names:
        img_path = os.path.join(current_dir, 'yolov5', 'custom_dataset', 'images', 'train', img_name)
        f.write(img_path)
        f.write('\n')
        
        
val_dir = os.path.join(current_dir, 'yolov5', 'custom_dataset', 'images', 'val')

img_names_val = os.listdir(val_dir)

with open('val.txt', 'w') as fw:
    for img_name in img_names_val:
        img_path = os.path.join(current_dir, 'yolov5', 'custom_dataset', 'images', 'val', img_name)
        fw.write(img_path)
        fw.write('\n')


#test_dir = os.path.join(current_dir, 'yolov5', 'custom_dataset', 'images', 'test')

#img_names_test = os.listdir(test_dir)

#with open('test.txt', 'w') as fw:
#    for img_name in img_names_test:
#        img_path = os.path.join(current_dir, 'yolov5', 'custom_dataset', 'images', 'test', img_name)
#        fw.write(img_path)
#        fw.write('\n')


