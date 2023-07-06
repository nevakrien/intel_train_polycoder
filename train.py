import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
from tqdm import tqdm
import os


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, file_path,pad_token=1):
        self.data = np.load(file_path)
        self.pad_token=1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        labels = torch.IntTensor(self.data[idx])
        mask=labels==-100 
        input_ids=labels.clone()
        input_ids[mask]=self.pad_token
        labels=labels.to(torch.long)
        return input_ids,labels,mask
    

def train(model, train_loader, test_loader, optimizer, scheduler, num_iters, save_interval, eval_interval, checkpoint_dir,device='cpu'):
    model.train()
    print(f'\ncomputations are done on {device}\n')


    # Training Loop
    for epoch in range(1, num_iters+1):
        print(f'Epoch {epoch}/{num_iters}')
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0
        total_corect=0
        total_seen=0

        # Training loop with tqdm
        train_loader_tqdm = tqdm(train_loader)
        for batch in train_loader_tqdm:
            # Zero the gradients
            optimizer.zero_grad()

            batch=(x.to(device) for x in batch)            
            input_ids,labels,mask=batch    
            
            # Forward pass
            outputs = model(input_ids, labels=labels,attention_mask=mask)
            
            # Get the loss and logits from the outputs
            loss = outputs.loss
            logits = outputs.logits            
            
           # Calculate accuracy
            idx=~mask.bool()
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions = (predictions[idx] == labels[idx]).sum()
            num_preds = idx.sum()
            assert num_preds>0 

            #metrics and logs
            total_loss += loss.cpu().detach().item()
            total_corect+=correct_predictions.cpu().detach().item()
            total_seen+=num_preds.cpu().detach().item()

            # Backward pass
            loss.backward() 
            optimizer.step()
            scheduler.step()


            train_loader_tqdm.set_postfix({'training_loss': total_loss /  (train_loader_tqdm.n + 1),'training_accuracy': total_corect /total_seen})

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_corect/total_seen
        print(f"Average training loss: {avg_train_loss}")
        print(f"Average training accuracy: {avg_train_accuracy}")

        # Evaluation
        if epoch % eval_interval == 0 or epoch==num_iters:
            model.eval()
            total_loss = 0
            total_corect=0
            total_seen=0

            # Adding tqdm to evaluation loop
            test_loader_tqdm = tqdm(test_loader, desc="Evaluating")
            for batch in test_loader_tqdm:
                with torch.no_grad():
                    # Load data and labels
                    batch=(x.to(device) for x in batch)
                    input_ids,labels,mask=batch    
                    
                    # Forward pass
                    outputs = model(input_ids, labels=labels,attention_mask=mask)
                    
                    # Get the loss and logits from the outputs
                    loss = outputs.loss
                    logits = outputs.logits            
                    
                   # Calculate accuracy
                    idx=~mask.bool()
                    predictions = torch.argmax(logits, dim=-1)
                    correct_predictions = (predictions[idx] == labels[idx]).sum()
                    num_preds = idx.sum()
                    assert num_preds>0 

                    #metrics and logs
                    total_loss += loss.cpu().detach().item()
                    total_corect+=correct_predictions.cpu().detach().item()
                    total_seen+=num_preds.cpu().detach().item()

                    # Update the progress bar
                    test_loader_tqdm.set_postfix({'eval_loss': total_loss / (test_loader_tqdm.n + 1),'eval_accuracy': total_corect/total_seen})

            # Calculate the average loss over the training data.
            avg_eval_loss = total_loss / len(test_loader)
            avg_eval_accuracy = total_corect/total_seen
            print(f"Average eval loss: {avg_eval_loss}")
            print(f"Average eval accuracy: {avg_eval_accuracy}")
            


            # Save a checkpoint
            if epoch % save_interval == 0 or epoch==num_iters:
                print(f'saving at: {checkpoint_dir}/checkpoint_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_train_loss,
                }, f'{checkpoint_dir}/checkpoint_{epoch}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train GPT-2 model")

    parser.add_argument('--config', type=str, default='./configs/160m', help="Path to the config for building the model")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the train and test data")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.00016, help="Learning rate for the optimizer")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs to train")
    parser.add_argument('--save_interval', type=int, default=1, help="Interval to save model checkpoints")
    parser.add_argument('--eval_interval', type=int, default=1, help="Interval to evaluate the model on test data")
    parser.add_argument('--xpu', type=int, default=0, help="required for runing on intel fast")
    #parser.add_argument('--num_xpus', type=int, default=None, help="required for runing on intel fast")

    args = parser.parse_args()
    if args.xpu:
        import intel_extension_for_pytorch as ipex
        device='xpu'
    else:
        device='cpu'

    config=GPTNeoXConfig.from_pretrained(args.config)
    model=GPTNeoXForCausalLM(config)

    optimizer = AdamW(model.parameters(), lr=0.00016, betas=(0.9, 0.999), eps=1.0e-8)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1, eta_min=0, last_epoch=-1) 
    
    train_data_path = os.path.join(args.data_dir, 'train_tokens.npy')
    test_data_path = os.path.join(args.data_dir, 'test_tokens.npy')

    train_dataset = TextDataset(train_data_path)
    test_dataset = TextDataset(test_data_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    train(model, train_loader, test_loader, optimizer, scheduler, args.epochs, args.save_interval, args.eval_interval, args.save_dir,device)
