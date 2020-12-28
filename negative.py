import json 
import os 
import random 


'''
data structure:
1. input: cadidates to be score
2. caption: correct image caption 
3. label: 0 or 1  
'''

# change list into one string 
def sentence_combine(caption_list):
    caption = ''
    for word in caption_list:
        caption += word + ' '
    # print(caption)
    return caption.strip()


# random repetion one token for a caption 
def repetition(caption_dict):
    caption = caption_dict['caption'].split()
    repetition_position = random.randint(1,len(caption)-1) 
    caption[repetition_position] = caption[repetition_position-1] 
    caption = sentence_combine(caption)
    caption_dict['input'] = caption 
    return caption_dict


# subtition one token with antonym
def substitution(caption_dict):
    return sentence_combine(caption_dict)


# disturbance with vistual attack 
def disturbance(caption_dict):
    return sentence_combine(caption_dict)


# generate negative samples for metric training 
def negative_create():
    data_path = 'data'
    annotations = os.path.join(data_path, 'annotations')
    annotations_path = os.path.join(annotations, 'captions_train2014.json')
    with open(annotations_path, 'r', encoding='utf-8') as j:
        caption = json.load(j)
    caption = caption['annotations']
    train_data = []
    for i in range(100):
        caption_dict = caption[i]
        if random.random() < 0.5:
            caption_dict = repetition(caption_dict)
            caption_dict['label'] = 0
        else:
            caption_dict['input'] = caption_dict['caption']
            caption_dict['label'] = 1
        train_data.append(caption_dict)
    print(train_data)
    with open(os.path.join(data_path, 'train_data.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)


if __name__ == "__main__":
    negative_create() 
