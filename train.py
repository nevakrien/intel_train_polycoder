import argparse
import torch
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
from tqdm import tqdm
import os
import json


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.x=data['x']
        self.y=data['y']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        x = torch.tensor(self.x[idx],dtype=torch.int32)
        y = torch.tensor(self.y[idx],dtype=torch.int64)
        mask=y!=-100
        return x,y,mask

def get_metrics(model,batch,mini_batch_size,device,do_grads=False): 
    batch_size=batch[0].size(0)
    ans={'loss':0,'num_preds':0,'corect':0}
    for i in range(0,batch_size,mini_batch_size):
        
        mini_batch=[x[i:i+mini_batch_size] for x in batch]
        #print([x.shape for x in mini_batch])
        
        mini_metrics=get_min_batch_metrics(model,mini_batch,device,do_grads)
        for k,v in mini_metrics.items():
            ans[k]+=v
    return ans

def get_min_batch_metrics(model,batch,device,do_grads):
    batch=(x.to(device) for x in batch)            
    x,y,mask=batch
    try:
        logits = model(x,mask).logits
    except Exception as e:
        print(mask)
        print(x)
        raise e

    num_preds = mask.sum()
    y=y[mask]
    logits=logits[mask]
    
    loss=F.cross_entropy(logits, y)
    if do_grads:
        loss.backward() 

    preds=torch.argmax(logits,dim=-1) 
    corect=(preds==y).sum()

    return {'loss':loss,'num_preds':num_preds,'corect':corect}
    
def train(model, train_loader, test_loader,mini_batch_size, optimizer, scheduler, num_iters, 
    save_interval, eval_interval, checkpoint_dir,device,
    train_denominator, test_denominator):
    model.train()
    print(f'\ncomputations are done on {device}\n')


    # Training Loop
    for epoch in range(1, num_iters+1):
        print(f'Epoch {epoch}/{num_iters}')
        model.train()
        # Reset the total loss for this epoch.
        total_loss=0
        total_corect=0
        total_seen=0

        # Training loop with tqdm
        train_loader_tqdm = tqdm(train_loader)
        for batch in train_loader_tqdm: 

            optimizer.zero_grad()

                
            
            metrics=get_metrics(model,batch,mini_batch_size,device,do_grads=True)
            loss=metrics['loss']
            num_preds=metrics['num_preds']
            corect=metrics['corect']
            
            total_loss += (loss*num_preds).cpu().detach().item()
            total_corect+=corect.cpu().detach().item()
            total_seen+=num_preds.cpu().detach().item()

            # Backward pass
            optimizer.step()
            scheduler.step()


            train_loader_tqdm.set_postfix({'training_loss':total_loss/total_seen,'training_accuracy': total_corect /total_seen})

        # Calculate the average loss over the training data.
        avg_train_loss= total_loss/total_seen
        avg_train_accuracy = total_corect/total_seen
        train_perplexity=np.exp(total_loss/train_denominator)

        print(f"Train Perplexity: {train_perplexity}")
        print(f"Average training loss: {avg_train_loss}")
        print(f"Average training accuracy: {avg_train_accuracy}")

        # Evaluation
        if epoch % eval_interval == 0 or epoch==num_iters:
            model.eval()
            total_loss=0
            total_corect=0
            total_seen=0

            # Adding tqdm to evaluation loop
            test_loader_tqdm = tqdm(test_loader, desc="Evaluating")
            for batch in test_loader_tqdm:
                with torch.no_grad():
                           
                    
                    metrics=get_metrics(model,batch,mini_batch_size,device)
                    loss=metrics['loss']
                    num_preds=metrics['num_preds']
                    corect=metrics['corect']
                    
                    total_loss += (loss*num_preds).cpu().detach().item()
                    total_corect+=corect.cpu().detach().item()
                    total_seen+=num_preds.cpu().detach().item()

                    # Update the progress bar
                    test_loader_tqdm.set_postfix({'eval_loss':total_loss/total_seen,'eval_accuracy': total_corect/total_seen})

            # Calculate the average loss over the training data.
            avg_eval_loss= total_loss/total_seen
            avg_eval_accuracy = total_corect/total_seen
            eval_perplexity=np.exp(total_loss/train_denominator)

            print(f"Eval Perplexity: {eval_perplexity}")
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
                    'train perplexity': train_perplexity,
                    'eval loss':avg_eval_loss,
                    'eval perplexity':eval_perplexity
                }, f'{checkpoint_dir}/checkpoint_{epoch}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train GPT-2 model")

    parser.add_argument('--config', type=str, default='./configs/160m', help="Path to the config for building the model")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the train and test data")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument('--batch_size', type=int, default=2, help="Big batch sizes are alowed")
    parser.add_argument('--mini_batch_size', type=int, default=0, help="should match to ur machine")
    parser.add_argument('--lr', type=float, default=0.00016, help="Learning rate for the optimizer")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs to train")
    parser.add_argument('--save_interval', type=int, default=1, help="Interval to save model checkpoints")
    parser.add_argument('--eval_interval', type=int, default=1, help="Interval to evaluate the model on test data")
    parser.add_argument('--xpu', type=int, default=0, help="required for runing on intel fast")
    #parser.add_argument('--num_xpus', type=int, default=None, help="required for runing on intel fast")

    args = parser.parse_args()

    if args.mini_batch_size==0:
        args.mini_batch_size=args.batch_size
    
    if args.xpu:
        import intel_extension_for_pytorch as ipex
        device='xpu'
    else:
        device='cpu'

    config=GPTNeoXConfig.from_pretrained(args.config)
    model=GPTNeoXForCausalLM(config).to(device)

    optimizer = AdamW(model.parameters(), lr=0.00016, betas=(0.9, 0.999), eps=1.0e-8)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1, eta_min=0, last_epoch=-1) 
    
    pygments_path=os.path.join(args.data_dir,'overhead.json')
    train_data_path = os.path.join(args.data_dir, 'train_tokens.npz')
    test_data_path = os.path.join(args.data_dir, 'test_tokens.npz')

    with open(pygments_path) as f:
        pygments_vocab=json.load(f)['vocab']
    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)

    train_denominator=pygments_vocab*sum(train_data['pygments_lens'])
    test_denominator=pygments_vocab*sum(test_data['pygments_lens'])

    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    train(model, train_loader, test_loader,args.mini_batch_size, optimizer, scheduler,
     args.epochs, args.save_interval, args.eval_interval, args.save_dir,device,
     train_denominator,test_denominator)
