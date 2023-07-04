"""
tokenize_data.py: 
Tokenizes and splits a text dataset into training and testing sets, saving each set to a numpy file.
"""

import os
import json
import tempfile 
import hashlib

import numpy as np
from transformers import GPT2Tokenizer
from multiprocessing import cpu_count, Pool


def get_hash(code):
    hash = hashlib.sha256(code.encode('UTF-8'))
    return hash.hexdigest()

def load_data(file):
    data = {}
    errors = 0
    with open(file,'rb') as f:
        for i,line in enumerate(f):
            try:
                t = json.loads(line)['content']
                k = get_hash(t)
                data[k] = t
            except Exception as e:
                print(f'Errored on file {file} at line {i}: {e}')
                errors += 1
    return data, errors

def load_all_data(dir_path, num_workers: int = None):
    num_workers = num_workers if num_workers else cpu_count()
    
    print("Starting loading data...")
    with Pool(num_workers) as p:
        results = p.map(load_data, [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith('.json')])
    print("Finished loading data.")
    
    data = {}
    total_errors = 0
    for result in results:
        data.update(result[0])
        total_errors += result[1]
    print(f"Loaded data from {len(results)} files with a total of {total_errors} errors.")
    
    return list(data.values())

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
    pad_token_id = tokenizer.pad_token_id

    if len(tokens) > gpt_cut:
        chunks = [tokens[i : i + max_len] for i in range(0, len(tokens), max_len)]
        # Pad the last chunk to max_len with the pad_token_id
        chunks[-1] = chunks[-1] + [pad_token_id]*(max_len - len(chunks[-1]))
        return chunks
    else:
        return None


def preprocess_data(dir_path, save_path, tokenizer_path, max_len, gpt_cut, mem_cut, test_size,debug_cut_size):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    codes = load_all_data(dir_path)

    if debug_cut_size!=None:
        codes=codes[:debug_cut_size]
        print(f'cuted codes to length{len(codes)}')

    print("Starting tokenization...")
    with Pool(cpu_count()) as p:
        token_chunks = p.map(tokenize_text, [(text, tokenizer, max_len, gpt_cut, mem_cut) for text in codes])

    # Filter out None results from tokenization
    tokens = []
    for chunks in token_chunks:
        if chunks is not None:  # Filter out None results
            for chunk in chunks:
                tokens.append(chunk)  # Flatten the list
    print(f"Finished tokenization. Kept {len(tokens)} sequences.")
    # Convert to numpy array
    tokens = np.array(tokens)

    # Split into train and test sets
    indices = np.arange(len(tokens))
    test_indices = np.random.choice(indices, size=test_size, replace=False)
    train_indices = np.array(list(set(indices) - set(test_indices)))

    train_tokens = tokens[train_indices]
    test_tokens = tokens[test_indices]

    # Save to numpy files
    np.save(os.path.join(save_path, "train_tokens.npy"), train_tokens)
    np.save(os.path.join(save_path, "test_tokens.npy"), test_tokens)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", help="Directory containing the data files", required=True)
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
    preprocess_data(args.dir_path, args.save_path, args.tokenizer_path, args.max_len, args.gpt_cut, args.mem_cut, args.test_size, args.debug_cut_size) 