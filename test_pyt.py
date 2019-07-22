import torch
import argparse
import re
import os
import copy
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)



"""
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
assert len(encoded_layers) == 12

print(encoded_layers)
print(len(encoded_layers))

"""

def load_model(modeldir):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(modeldir)

    # Load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained(modeldir)
    model.eval()
    model.to('cuda')
    return model,tokenizer


def load_modelGPT():
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    # Load pre-trained model (weights)
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    model.eval()
    model.to('cuda')
    return model,tokenizer

def prep_input(input_sents, tokenizer,bert=True):
    for sent in input_sents:
        masked_index = None
        text = []
        mtok = '[MASK]'
        if not bert:
            sent = re.sub('\[MASK\]','X',sent)
            mtok = 'x</w>'
        if bert: text.append('[CLS]')
        text += sent.strip().split()
        if text[-1] != '.': text.append('.')
        if bert: text.append('[SEP]')
        text = ' '.join(text)
        #print('\n')
        tokenized_text = tokenizer.tokenize(text)
        print(tokenized_text)
        masked_index = [i for i,tok in enumerate(tokenized_text) if tok == mtok]
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        yield tokens_tensor, masked_index,tokenized_text

def mix_weights(pretrained_dir,finetuned_dir):
    #initialize the actual model we want: BertForMaskedLM. that will have all the structure we need
    #load it with the pretrained weights. that will fill the top as needed

    tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
    model = BertForMaskedLM.from_pretrained(pretrained_dir)
    # state_dict_pt = torch.load('/scratch/BERT/params/bert-base-uncased/pytorch_model.bin')
    #load the state dict with the fine-tuned weights
    sd_finetuned = torch.load(os.path.join(finetuned_dir,'pytorch_model.bin'))
    #loop through all weights in the state dict -- change names if needed (add / remove bert. ?)
    sd_new = copy.deepcopy(model.state_dict())
    sd_pt = copy.deepcopy(model.state_dict())
    # print(model.state_dict()['bert.encoder.layer.11.output.dense.weight'])
    # print('\n\n')
    # print(model.state_dict()['cls.predictions.transform.dense.weight'])
    # print('\n\n')
    # print(model.state_dict()['bert.embeddings.word_embeddings.weight'])
    for name,param in sd_finetuned.items():
        if name in sd_new:
            print(name)
            param = param.data
            sd_new[name] = copy.deepcopy(param)
        else:
            print('NOT COPIED from fine-tuned: %s'%name)
    model.load_state_dict(sd_new)
    model.eval()
    model.to('cuda')
    return model,tokenizer


def get_predictions(input_sents,model,tokenizer,k=5,bert=True):
    token_preds = []
    tok_probs = []
    for tokens_tensor, masked_index,_ in prep_input(input_sents,tokenizer,bert=bert):
        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        # segments_tensors = segments_tensors.to('cuda')
        # Predict all tokens
        with torch.no_grad():
            predictions = model(tokens_tensor)
        predicted_tokens = []
        predicted_token_probs = []
        print(predictions.size())
        for mi in masked_index:
            if bert:
                softpred = torch.softmax(predictions[0,mi],0)
            else:
                softpred = torch.softmax(predictions[0, mi, :],0)
            top_inds = torch.argsort(softpred,descending=True)[:k].cpu().numpy()
            top_probs = [softpred[tgt_ind].item() for tgt_ind in top_inds]
            top_tok_preds = tokenizer.convert_ids_to_tokens(top_inds)
            if not bert:
                top_tok_preds = [re.sub('\<\/w\>','',e) for e in top_tok_preds]
            # torch.gather(predictions[0,mi],0,top_inds)
            #predicted_index = torch.argmax(predictions[0, mi]).item()
            #this_predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
            predicted_tokens.append(top_tok_preds)
            predicted_token_probs.append(top_probs)
        # else:
        #     predicted_index = torch.argmax(predictions[0, -1, :]).item()
        #     predicted_tokens = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        # print(tokens_tensor)
        # print(predicted_tokens)
        token_preds.append(predicted_tokens)
        tok_probs.append(predicted_token_probs)
    return token_preds,tok_probs

def get_probabilities(input_sents,tgtlist,model,tokenizer,bert=True):
    token_probs = []
    oov_list = []
    for i,(tokens_tensor, masked_index,_) in enumerate(prep_input(input_sents,tokenizer,bert=bert)):
        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        # segments_tensors = segments_tensors.to('cuda')
        # Predict all tokens
        with torch.no_grad():
            predictions = model(tokens_tensor)
        pred_tuple = []
        tgt = tgtlist[i]
        # print(input_sents[i])
        # print(tgt)
        # print(masked_index)
        oov = False
        for mi in masked_index:
            if bert:
                softpred = torch.softmax(predictions[0,mi],0)
            else:
                softpred = torch.softmax(predictions[0, mi, :],0)
            try:
                tgt_ind = tokenizer.convert_tokens_to_ids([tgt])[0]
            except:
                this_tgt_prob = np.nan
                oov = True
            else:
                this_tgt_prob = softpred[tgt_ind].item()
            pred_tuple.append(this_tgt_prob)
        # print(pred_tuple)
        if oov == True: oov_list.append(tgt)
        token_probs.append(pred_tuple)
        # print(pred_tuple)
    return token_probs, oov_list

def get_attention(input_sents,model,tokenizer,bert=True):

    for tokens_tensor, masked_index, tokenized_text in prep_input(input_sents,tokenizer,bert=bert):
        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        # segments_tensors = segments_tensors.to('cuda')
        # Predict all tokens
        with torch.no_grad():
            _,attention = model(tokens_tensor,output_all_encoded_layers=True)
            for layer,lyr_atten in enumerate(attention):
                print('LAYER %s\n'%layer)
                print('\n')
                for head,head_atten in enumerate(lyr_atten[0]):
                    print('HEAD %s\n'%head)
                    for word_ind,word_atten in enumerate(head_atten):
                        print(tokenized_text[word_ind])
                        print([tokenized_text[e] for e in torch.argsort(word_atten,descending=True)[:1].cpu().numpy()])
                    print('\n')
                print('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str, required=True)
    args = parser.parse_args()
    get_predictions(args.input_file)
