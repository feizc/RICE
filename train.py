from transformers import * 
import torch 
import os 

model_path = 'ckpt'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased') 

torch.save(model.state_dict(), os.path.join(model_path, 'pytorch_model.bin'))
model.config.to_json_file(os.path.join(model_path, 'config.json'))
tokenizer.save_vocabulary(model_path)