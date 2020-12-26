from transformers import * 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss 
import os 
import torch 


class RICE(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2 
        self.bert =  BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2) 
        
        self.rec_linear = nn.Linear(config.hidden_size, config.vocab_size)
        self.image_ff = nn.Linear(2048, config.hidden_size)
        self.image_inverse_ff = nn.Linear(config.hidden_size, 2048)
        
        self.init_weights() 
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.rec_linear, self.bert.embeddings.word_embeddings)
    
    def forward(self, input_embs, token_type_ids=None, target_caps=None, labels=None, reconstruct_target=None):
        outputs = self.bert(
            inputs_embeds=input_embs,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) 

        # classification loss 
        if labels is not None: 
            cross_loss_fct = CrossEntropyLoss()
            loss = cross_loss_fct(logits, labels) 
            return loss 
        
        # reconstruction loss 

        return logits 


if __name__ == "__main__":
    config = BertConfig()
    model = RICE(config)
    input = torch.randn(1, 6, 768)
    output = model(input)
    print(output)
