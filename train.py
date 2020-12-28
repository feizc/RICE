from transformers import * 
import torch 
import os 
from model import RICE 
from dataset import RiceDataset 


SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[IMG]', '[TXT]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[IMG]', '[TXT]'], 'pad_token':'[PAD]'}


def train(): 
    # hyper parameter setting 
    model_path = 'ckpt' 
    data_path = 'data'
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu') 
    epochs = 10 
    lr = 1e-4 
    gradient_accumulation_steps = 5 

    # model initilization 
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = RICE.from_pretrained(model_path) 
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
    model.resize_token_embeddings(len(tokenizer))

    model = model.to(device) 
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # dataset 
    dataset = RiceDataset(data_path, tokenizer)

    model.train()
    for epoch in range(epochs): 
        iteration = 1 
        for instance in dataset:
            img, input_cap, target_cap, label = instance 
            img, input_cap, target_cap, label = img.to(device), input_cap.to(device), target_cap.to(device), label.to(device) 
            img_embs = model.image_ff(img) 
            input_embs = model.bert.embeddings.word_embeddings(input_cap) 
            input_embs = torch.cat([img_embs, input_embs], dim=0)
            # print(img_embs.size(), input_embs.size()) 
            loss = model(input_embs.view(1, -1, 768), labels=label) 
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if iteration % gradient_accumulation_steps == 0: 
                optimizer.step()
                optimizer.zero_grad()

            print(loss.item()) 
            break
        break

    '''
    torch.save(model.state_dict(), os.path.join(model_path, 'pytorch_model.bin'))
    model.config.to_json_file(os.path.join(model_path, 'config.json'))
    tokenizer.save_vocabulary(model_path) 
    '''


if __name__ == "__main__":
    train() 