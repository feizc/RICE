# RICE： reference-free image captioning evaluation

### Image caption evaluation 

structure: Cross-modality BERT 
input: image + caption candidate 
output: score 


### Negative sampling

1. Repetiton 
2. Substitution 
3. Visual attack 

# Coding Framework 

1. negative.py   generate negative samples for model training 
2. train.py   training the RICE model 
3. inference.py  given image-text pair, utilize the checkpoint to meature the quality 


