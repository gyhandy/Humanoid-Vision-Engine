import numpy as np
import json

with open('shape.json','r') as f:
    shape = json.load(f)

with open('texture.json','r') as f:
    texture = json.load(f)

with open('color.json','r') as f:
    color = json.load(f)

vector_dict = np.load('vector.npy',allow_pickle=True)
vector_dict = vector_dict.item()

label2name_dict = {'n02389026':'horse','n02391049':'zebra','n02129604':'tiger','n02510455':'panda',
                    'n02056570':'penguin','n04515003':'piano_keys','n02130308':'cheetah','n02117135':'hyena',
                    'n02100583':'dog','n07749582':'lemon','n01882714':'koala','n03404251':'fur','n02356798':'squirrel',
                    'n02119022':'fox','n02325366':'rabbit','n01855672':'goose','n02415577':'sheep','n02077923':'sea_lion',
                    'n02504458':'elephant','n02444819':'otter','n01847000':'duck','n01514668':'cock','n02481823':'chimpanzee',
                    'n02412080':'goat','n07747607':'orange','n02802426':'ball','n02403003':'bull','red':'apple','tomato':'tomato',
                    'cherry':'cherry','pear':'pear','turkey':'turkey','seal':'seal','porpoise':'porpoise','alpaca':'alpaca',
                    'pigeon':'pigeon','fowl':'fowl','n01828970':'bird','n02114367':'wolf','n02129165':'lion','n02132136':'bear','donkey':'donkey'}

candidate_list = ['fowl','zebra','bear','wolf','sheep','seal','apple','train','bag','balloon','car','pen','table','eagle']
test_dict = {'fowl':0,'zebra':1,'lion':2,'bear':3,'wolf':4,'sheep':5,'seal':6,'apple':7}
candidate_vector = []
for candidate in candidate_list:
    candidate_vector.append(vector_dict[candidate])
candidate_vector = np.array(candidate_vector)

right = np.zeros(len(test_dict))
total = np.zeros(len(test_dict))

shape_rank = {}
texture_rank = {}
color_rank = {}

confusion_matrix = {}
for image_name in shape:
    class_name = label2name_dict[image_name.split('_')[0]]
    if not class_name in confusion_matrix.keys():
        confusion_matrix[class_name] = {}

    if not class_name in shape_rank.keys():
        shape_rank[class_name] = {}
        texture_rank[class_name] = {}
        color_rank[class_name] = {}
    classes_shape = shape[image_name]['class']
    classes_texture = texture[image_name]['class']
    classes_color = color[image_name]['class']

    for class_shape in classes_shape[:3]:
        shape_name = label2name_dict[class_shape]
        if not shape_name in shape_rank[class_name]:
            shape_rank[class_name][shape_name] = 1
        else:
            shape_rank[class_name][shape_name] = shape_rank[class_name][shape_name] + 1
    
    for class_texture in classes_texture[:3]:
        texture_name = label2name_dict[class_texture]
        if not texture_name in texture_rank[class_name]:
            texture_rank[class_name][texture_name] = 1
        else:
            texture_rank[class_name][texture_name] = texture_rank[class_name][texture_name] + 1

    for class_color in classes_color[:3]:
        color_name = label2name_dict[class_color]
        if not color_name in color_rank[class_name]:
            color_rank[class_name][color_name] = 1
        else:
            color_rank[class_name][color_name] = color_rank[class_name][color_name] + 1

    seed_list = classes_shape[0:3]+classes_texture[0:2]+classes_color[0:1]
    for i in range(len(seed_list)):
        seed_list[i] = label2name_dict[seed_list[i]]
    seed_vector = []
    for seed in seed_list:
        seed_vector.append(vector_dict[seed])
    seed_vector = np.array(seed_vector)

    result = None
    score = -np.inf

    for i in range(len(candidate_vector)):
        candidate_vector_image = candidate_vector[i]
        score_candidate = np.dot(candidate_vector_image, seed_vector.T)
        denom = np.linalg.norm(candidate_vector) * np.linalg.norm(seed_vector, axis=1)
        score_all = score_candidate/denom
        score_all[np.isneginf(score_all)] = 0
        score_all = 0.5+0.5*score_all
        score_image = np.sum(score_all)
        if score_image > score:
            score = score_image
            result = candidate_list[i]
    total[test_dict[class_name]] += 1
    if result == class_name:
        right[test_dict[class_name]] += 1
    if not result in confusion_matrix[class_name].keys():
        confusion_matrix[class_name][result] = 1
    else:
        confusion_matrix[class_name][result] = confusion_matrix[class_name][result] + 1

for key in shape_rank.keys():
    rank = shape_rank[key]
    shape_rank[key] = sorted(shape_rank[key].items(),key=lambda item:item[1],reverse=True)

for key in texture_rank.keys():
    rank = texture_rank[key]
    texture_rank[key] = sorted(texture_rank[key].items(),key=lambda item:item[1],reverse=True)

for key in color_rank.keys():
    rank = color_rank[key]
    color_rank[key] = sorted(color_rank[key].items(),key=lambda item:item[1],reverse=True)

for key in confusion_matrix.keys():
    matrix = confusion_matrix[key]
    confusion_matrix[key] = sorted(confusion_matrix[key].items(),key=lambda item:item[1],reverse=True)

rank = {'shape':shape_rank,'texture':texture_rank,'color':color_rank}
rank = json.dumps(rank, indent=4)

confusion_matrix = json.dumps(confusion_matrix,indent=4)
with open('rank.json','w') as f:
    f.write(rank)

with open('confusion_matrix.json','w') as f:
    f.write(confusion_matrix)
print(right)
print(total)
print(right/total)
# print(shape_rank)
    

