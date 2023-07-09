"""
tokenize_data.py: 
Tokenizes and splits a text dataset into training and testing sets, saving each set to a numpy file.
"""

import os
import json
import csv
import tempfile 
import hashlib
from collections import defaultdict

import numpy as np
from transformers import GPT2Tokenizer
from multiprocessing import cpu_count, Pool


def get_hash(code):
    hash = hashlib.sha256(code.encode('UTF-8'))
    return hash.hexdigest()

def load_data(file):
    data = defaultdict(lambda:{})
    errors = 0
    with open(file,'rb') as f:
        for i,line in enumerate(f):
            try:
                repo=json.loads(line)
                t = repo['content']
                k = get_hash(t)
                #data[k] = t
                entry=data[k]
                if len(entry)==0:
                    entry.update({'text':t,'repos':[repo['repo_name']]})
                else: 
                    entry['repos'].append(repo['repo_name'])
            except Exception as e:
                print(f'Errored on file {file} at line {i}: {e}')
                errors += 1
    return dict(data), errors

def load_all_data(data_path,save_path, num_workers: int = None):
    num_workers = num_workers if num_workers else cpu_count()
    
    print("Starting loading data...")
    with Pool(num_workers) as p:
        results = p.map(load_data, [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.json')])
    print("Finished loading data.")

    data = defaultdict(lambda:{})
    total_errors = 0
    for result in results:
        total_errors += result[1]
        #updating the values
        for k,v in result[0].items():
            entry=data[k]
            if len(entry)==0:
                entry.update(v)
            else: 
                entry['repos'].extend(v['repos'])


    print(f"Loaded data from {len(results)} files with a total of {total_errors} errors.")

    return [x['text'] for x in data.values()],[x['repos'] for x in data.values()]

def get_mem_usage(code):
    '''
    get the memory usage of a file containing @param:code
    '''
    with tempfile.NamedTemporaryFile() as temp_file:
        file_path = temp_file.name

        temp_file.write(bytes(code, 'utf-8'))
        temp_file.flush()

        mem_usage = os.path.getsize(file_path)
        #print(mem_usage)

    return mem_usage 

def tokenize_text(params):
    text, tokenizer, max_len, gpt_cut, mem_cut = params

    if get_mem_usage(text) > mem_cut:
        return None

    tokens = tokenizer.encode(text)
    #this is bugged!!! pad_token_id = tokenizer.pad_token_id

    if len(tokens) > gpt_cut:

        x_chunks = [tokens[i : i + max_len] for i in range(0, len(tokens)-1, max_len)]
        x_chunks[-1] = x_chunks[-1] + [1]*(max_len - len(x_chunks[-1]))

        y_chunks = [tokens[i : i + max_len] for i in range(1, len(tokens), max_len)]
        y_chunks[-1] = y_chunks[-1] + [-100]*(max_len - len(y_chunks[-1]))
        return x_chunks,y_chunks
    else:
        return None

def save_numpy(token_chunks,path):
    x_chunks=[chunks[0] for chunks in token_chunks if chunks is not None]
    y_chunks=[chunks[1] for chunks in token_chunks if chunks is not None]

    #flattening the chunks 
    x = []
    for chunks in x_chunks:
        for chunk in chunks:
            x.append(chunk) 

    y = []
    for chunks in y_chunks:
        for chunk in chunks:
            y.append(chunk) 

    x=np.array(x,dtype=np.int64)
    y=np.array(y,dtype=np.int64)

    np.savez(path,x=x,y=y)



def preprocess_data(data_path, save_path, tokenizer_path, max_len, gpt_cut, mem_cut, test_size,debug_cut_size):
    assert os.path.exists(save_path) 

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    codes,names = load_all_data(data_path,save_path)

    if debug_cut_size!=None:
        codes=codes[:debug_cut_size]
        names=names[:debug_cut_size]
        print(f'cuted codes to length{len(codes)}')


    print("Starting tokenization...")
    with Pool(cpu_count()) as p:
        token_chunks = p.map(tokenize_text, [(text, tokenizer, max_len, gpt_cut, mem_cut) for text in codes])

    names=[name for name,chunks in zip(names,token_chunks) if chunks is not None]
    token_chunks=[chunks for chunks in token_chunks if chunks is not None]
    
    print(f"Finished tokenization. Kept {len(token_chunks)} files.")

    print('Spliting the dataset')
    # Split into train and test sets
    indices = np.arange(len(token_chunks))
    test_indices = np.random.choice(indices, size=test_size, replace=False)
    train_indices = np.array(list(set(indices) - set(test_indices)))

    train_chunks=[token_chunks[i] for i in train_indices]
    train_names=[names[i] for i in train_indices]

    test_chunks=[token_chunks[i] for i in test_indices]
    test_names=[names[i] for i in test_indices]

    print('Saving lengths')
    test_lengths = [len(chunks[0]) for chunks in test_chunks]
    with open(os.path.join(save_path,'test_lengths.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[length] for length in test_lengths])

    train_lengths = [len(chunks[0]) for chunks in train_chunks]
    with open(os.path.join(save_path,'train_lengths.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[length] for length in train_lengths])
    
    print('Saving repo names')
    with open(os.path.join(save_path,'train_names.json'), 'w') as file:
        json.dump(train_names, file)
    with open(os.path.join(save_path,'test_names.json'), 'w') as file:
        json.dump(test_names, file)

    print('Saving repo tokens')
    save_numpy(train_chunks,path=os.path.join(save_path, "train_tokens.npz"))
    save_numpy(test_chunks,path=os.path.join(save_path, "test_tokens.npz"))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Directory containing the data files", required=True)
    parser.add_argument("--save_path", help="Path to save the numpy files", required=True)
    parser.add_argument("--tokenizer_path", default="./configs/tokenizer", help="Path to the tokenizer")
    parser.add_argument("--max_len", default=2048, type=int, help="Maximum length for each sequence")
    parser.add_argument("--gpt_cut", default=100, type=int, help="Cut-off for token length")
    parser.add_argument("--mem_cut", default=1_000_000, type=int, help="Cut-off for memory usage")
    parser.add_argument("--test_size", default=2, type=int, help="Size of the test set")
    parser.add_argument("--debug_cut_size", default=10, type=int, help="Size for debugging. If None, use full dataset.")
    args = parser.parse_args()

    if args.debug_cut_size!=None:
        print('you have ran this aplication in debug mode')
    preprocess_data(args.data_path, args.save_path, args.tokenizer_path, args.max_len, args.gpt_cut, args.mem_cut, args.test_size, args.debug_cut_size) 