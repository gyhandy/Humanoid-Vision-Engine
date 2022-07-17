from PIL import Image
import os

source = '/lab/tmpig23b/u/zhix/Human-AI-Interface/clver/Data_V3'
target = '/lab/tmpig23b/u/zhix/Human-AI-Interface/clver/Data_V3_crop'
class_list = ['Object_A','Object_B','Object_C']
for class_name in class_list:
    if not os.path.exists(os.path.join(target,class_name)):
        os.mkdir(os.path.join(target,class_name))
    for root, dirs, files in os.walk(os.path.join(source, class_name)):
        break
    for file in files:
        img = Image.open(os.path.join(root, file))
        img_crop = img.crop([img.size[0]/6,img.size[1]/6,img.size[0]*5/6,img.size[1]*5/6])
        img_crop.save(os.path.join(target, class_name, file))