import torch 
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig, GPT2Tokenizer

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm 
import os 

# Define the training params
num_iters = 11
eval_interval = 5  
save_interval = 10 
checkpoint_dir = './checkpoints' 
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
model_dir='./configs/150m'
max_len=30#None #none for setting it by model config 
batch_size=16

tokenizer=GPT2Tokenizer.from_pretrained('./configs/tokenizer')
config=GPTNeoXConfig.from_pretrained(model_dir)
model=GPTNeoXForCausalLM(config)

if max_len==None:
    max_len=config.max_position_embeddings 

#loading dummby data 
import json
data_file='cpp000000000302.json' 

data = []
errors=[]
faultys=[]
with open(data_file,'rb') as f:
    for i,line in enumerate(f):
        try:
          data.append(json.loads(line))
        except Exception as e:
          print(f'errored at {i}')
          errors.append(e)
          faultys.append(line)

# Now 'data' is a list of all the JSON objects in the file
print(f'data: {len(data)} errors: {len(errors)}') 

codes=[d['content'] for d in data[0:100] if 'content' in d.keys()]

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        for text in tqdm(texts):
            encodings = tokenizer(text, truncation=True,
                                  #padding='max_length',
                                  max_length=max_len)
            #encodings=tokenizer(text)
            self.inputs.append(encodings['input_ids'])
            self.targets.append(encodings['input_ids'])

    def __getitem__(self, idx):
        item = {"input_ids": torch.IntTensor(self.inputs[idx]), 
                "labels": torch.IntTensor(self.targets[idx])}
        return item

    def __len__(self):
        return len(self.inputs)

# Tokenize your dataset and create a PyTorch Dataset
dataset = TextDataset(codes, tokenizer)

# Split dataset into training and test set
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True)
    return {"input_ids": input_ids, "labels": labels}

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
model=torch.nn.DataParallel(model, device_ids=['cpu'])

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
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device).to(torch.long)
        
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
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device).to(torch.long)
                
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, f'{checkpoint_dir}/checkpoint_{epoch}.pt')
