import torch 
import json 
from transformers import * 
from torch.utils.data import Dataset 
import h5py 
import os 


class RiceDataset(Dataset): 
    def __init__(self, data_path, tokenizer): 
        self.tokenizer = tokenizer 
        self.img_features = h5py.File(os.path.join(data_path, 'coco_detections.hdf5'))
        # cap_path = os.path.join(data_path, 'annotations')
        cap_path = os.path.join(data_path, 'train_data.json')
        with open(cap_path, 'r', encoding='utf-8') as j:
            self.cap = json.load(j)

    def __getitem__(self, i):
        cap_dict = self.cap[i]
        img_id = str(cap_dict['image_id']) + '_features'
        img = torch.FloatTensor(self.img_features[img_id])
        caption = cap_dict['caption']
        caption = self.tokenizer.tokenize(caption)
        caption = self.tokenizer.convert_tokens_to_ids(caption)
        caption = torch.Tensor(caption).long()

        label = torch.Tensor([cap_dict['label']]).long()

        return img, caption, label  
    
    def __len__(self):
        return len(self.cap)



if __name__ == "__main__":
    path = 'data'
    tokenizer = BertTokenizer('ckpt/vocab.txt')
    dataset = RiceDataset('data', tokenizer)
    print(dataset[0])


