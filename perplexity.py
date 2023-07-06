import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig,GPT2Tokenizer
from tqdm import tqdm
import os 

from tqdm import tqdm

import pygments
from pygments.lexers import get_lexer_by_name
import torch.nn.functional as F 

def hack_input_length(model,MAX_INPUT_LEN):
	MAX_LEN=model.config.max_position_embeddings

	b = torch.ones(MAX_INPUT_LEN, MAX_INPUT_LEN,dtype=torch.bool)
	b = torch.tril(b)
	for i in range(MAX_INPUT_LEN):
	    if(i>MAX_LEN):
	        b[i][:i-MAX_LEN]=0
	b = b.unsqueeze(0).unsqueeze(0)

	for l in model.base_model.layers:
	    l.attention.bias=b.clone().to(l.attention.bias.device)

def back_to_docs(data):
    ans=[]
    curent=np.zeros([0],dtype=int)
    for seq in data:
        end_cond =-100 in seq
        
        if end_cond:
            seq=seq[np.where(seq != -100)[0]]
        
        curent=np.concatenate([curent,seq])
        
        if end_cond:
            ans.append(curent)
            curent=np.zeros([0],dtype=int)
    return ans

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx])[None]

def get_denominator(doc_tokens,lexer):
    lex_vocab = sum(len(v) for v in lexer.tokens.values())
    num_tok =0
    for x in doc_tokens:
        num_tok+=len(list(pygments.lex(tokenizer.decode(x), lexer))) 
    return num_tok*lex_vocab

@torch.no_grad()
def get_neg_log(input_ids, model, MAX_INPUT_LEN):
    MAX_LEN = model.config.max_position_embeddings

    x = input_ids[:, :MAX_INPUT_LEN]
    out = model(x)
    ans = F.cross_entropy(out.logits, F.one_hot(x, out.logits.shape[-1]).to(float), reduction='sum')

    start = MAX_INPUT_LEN
    end = 2 * MAX_INPUT_LEN - MAX_LEN

    while start < input_ids.shape[1]:
        cut_kv=[[z[:,:,-MAX_LEN:] for z in y] for y in out.past_key_values]
        x = input_ids[:, start:end]
        out = model(x)
        ans += F.cross_entropy(out.logits, F.one_hot(x, out.logits.shape[-1]).to(float), reduction='sum')
        start = end
        end += MAX_INPUT_LEN - MAX_LEN

    return ans

    return ans

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Argument Parser for the program')
	parser.add_argument('--config', type=str, default='./configs/160m', help='Path to the config file')
	parser.add_argument('--data_dir', type=str, default='./test_garbage', help='Path to the data directory in which we assume data is name test.npy')
	parser.add_argument('--tokenizer_path', type=str, default='./configs/tokenizer', help='Path to the tokenizer')
	parser.add_argument('--checkpoint', type=str, default='./test_garbage/checkpoint_5.pt', help='Path to the checkpoint file')
	parser.add_argument('--max_input_len', type=int, default=4096, help='Maximum input length')
	parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or xpu)')
	parser.add_argument('--lang', type=str, required=True, help='Device to use (cpu or xpu)')

	args = parser.parse_args()

	#this apears first as a check that the user inserted a valid lang string
	lexer = get_lexer_by_name(args.lang) 

	data = np.load(os.path.join(args.data_dir, 'test_tokens.npy'))
	print('loaded data')
	state=torch.load(args.checkpoint)['model_state_dict']
	print('loaded model') 

	config = GPTNeoXConfig.from_pretrained(args.config)
	model=GPTNeoXForCausalLM(config)
	model.load_state_dict(state)
	model=model.to(args.device).to(float)

	hack_input_length(model,args.max_input_len)

	
	tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
	print('loaded tokenizer')
	
	data=back_to_docs(data) 

	
	d=get_denominator(tqdm(data,desc='calculating num tokens'),lexer)
	
	dataset=TokenDataset(data)
	
	neg_log=0. 
	for x in tqdm(dataset,desc='calculating neg log prob'):
		neg_log+=get_neg_log(x,model,args.max_input_len)

	print(torch.exp(neg_log/d)
	print(torch.exp(neg_log/d).cpu().detach().item())
	print('yay')
