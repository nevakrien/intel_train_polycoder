import torch 
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig, GPT2Tokenizer

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm 

import os 
import json
import tempfile

from multiprocessing import Pool, cpu_count
from transformers import PreTrainedTokenizerFast
import hashlib


# Define the training params
num_iters = 11
eval_interval = 5  
save_interval = 10 
checkpoint_dir = './checkpoints' 
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
model_dir='./configs/150m'
max_len=None #none for setting it by model config 
debug_cut_size=10
batch_size=2
gpt_cut=100
mem_cut=1_000_000

tokenizer=GPT2Tokenizer.from_pretrained('./configs/tokenizer')
config=GPTNeoXConfig.from_pretrained(model_dir)
model=GPTNeoXForCausalLM(config)

if max_len==None:
    max_len=config.max_position_embeddings 

#loading dummby data 
data_file='cpp000000000302.json' 

def get_hash(code):
    hash = hashlib.sha256(code.encode('UTF-8'))
    return hash.hexdigest()

data = {}
errors=[]
faultys=[]
with open(data_file,'rb') as f:
    for i,line in enumerate(f):
        try:
          #data.append(json.loads(line))
          t=json.loads(line)['content'] 
          k=get_hash(t)
          data.update({k:t})
        except Exception as e:
          print(f'errored at {i}: {e}')
          errors.append(e)
          faultys.append(line)

# Now 'data' is a list of all the JSON objects in the file
print(f'data: {len(data)} errors: {len(errors)}') 

codes=[v for v in data.values()]

if debug_cut_size!=None:
    codes=codes[:debug_cut_size]
    print(f'cuted codes to length{len(codes)}')


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

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer: PreTrainedTokenizerFast, max_len, gpt_cut, mem_cut,num_workers: int = None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.gpt_cut = gpt_cut
        self.mem_cut=mem_cut
        self.num_workers = num_workers if num_workers else cpu_count()

        print("Starting tokenization...")
        with Pool(self.num_workers) as p:
            tokens = p.map(self.tokenize_text, texts)

        # Filter out None results from tokenization
        self.tokens = []
        for chunks in tokens:
            if chunks is not None:  # Filter out None results
                for chunk in chunks:
                    self.tokens.append(chunk)  # Flatten the list
        print(f"Finished tokenization. Kept {len(self.tokens)} sequences.")

        if debug_cut_size!=None:
            self.tokens=self.tokens[:debug_cut_size]
            print(f'cuting size for debuging purposes to {len(self.tokens)}')

    def tokenize_text(self, text):
        if get_mem_usage(text)>self.mem_cut:
            #print('mem fail')
            return None

        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.gpt_cut:
            #print('passed')
            return [tokens[i : i + self.max_len] for i in range(0, len(tokens), self.max_len)]
        else:
            #print('tokens fail')
            return None

    def __getitem__(self, idx):
        return torch.IntTensor(self.tokens[idx])

    def __len__(self):
        return len(self.tokens)




# Tokenize your dataset and create a PyTorch Dataset
dataset = TextDataset(codes, tokenizer,max_len, gpt_cut, mem_cut)
print(len(dataset))
# Split dataset into training and test set
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

def collate_fn(batch):
    return  pad_sequence(batch, batch_first=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Define the device
try: 
    device='xpu'
    model=model.to(device)
except:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model=model.to(device)
    #model=torch.nn.DataParallel(model, device_ids=['cpu'])
print(f'\ncomputations are done on {device}\n')

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00016, betas=(0.9, 0.999), eps=1.0e-8)


# Define the learning rate scheduler
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=num_iters, T_mult=1, eta_min=0, last_epoch=-1) 

model.train()
# Training Loop
for epoch in range(1, num_iters+1):
    print(f'Epoch {epoch}/{num_iters}')
    
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop with tqdm
    train_loader_tqdm = tqdm(train_loader)
    for batch in train_loader_tqdm:
        # Zero the gradients
        optimizer.zero_grad()

        # Load data and labels
        input_ids = batch.to(device)
        labels = input_ids.to(torch.long)
        
        # Forward pass
        outputs = model(input_ids, labels=labels)
        
        # Get the loss from the outputs
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Update the learning rate.
        scheduler.step()

        # Add the loss to the total loss
        total_loss += loss.cpu().detach().item()

        # Update the progress bar
        train_loader_tqdm.set_postfix({'running_loss': total_loss /  (train_loader_tqdm.n + 1)})

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_loader)
    print(f"Average training loss: {avg_train_loss}")

    # Evaluation
    if epoch % eval_interval == 0 or epoch==num_iters:
        model.eval()
        eval_total_loss = 0

        # Adding tqdm to evaluation loop
        test_loader_tqdm = tqdm(test_loader, desc="Evaluating")
        for batch in test_loader_tqdm:
            with torch.no_grad():
                # Load data and labels
                input_ids = batch.to(device)
                labels = input_ids.to(torch.long)
                
                # Forward pass
                outputs = model(input_ids, labels=labels)
                
                # Get the loss from the outputs
                loss = outputs.loss

                # Add the loss to the total loss
                eval_total_loss += loss.cpu().detach().item()

                # Update the progress bar
                test_loader_tqdm.set_postfix({'eval_loss': eval_total_loss / (test_loader_tqdm.n + 1)})

        avg_eval_loss = eval_total_loss / len(test_loader)
        print(f"Average evaluation loss: {avg_eval_loss}")
        model.train()


        # Save a checkpoint
        if epoch % save_interval == 0 or epoch==num_iters:
            print(f'saving at: {checkpoint_dir}/checkpoint_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, f'{checkpoint_dir}/checkpoint_{epoch}.pt')
