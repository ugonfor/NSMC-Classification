# import library
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F

from tqdm import tqdm
import argparse

# import my codes
from utils import set_seed, save_model, load_model
from preprocess import NSMCDataset
from model import BertClassifier

parser = argparse.ArgumentParser(description='device info')
parser.add_argument('--device', type=str)
args = parser.parse_args()


# configure
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.device:
    device = args.device

MULTIGPU = False 
MAX_LEN = 30
BATCH_SIZE = 32
MODE = 'train' # 'load'
model_path = './model_pt'
EPOCHS = 4

logfile = "./log"

log = open(logfile, "w")

set_seed(42)


def TrainOneEpoch(model, optimizer, data_loader, device, epoch):
    model.train()
    model.zero_grad()

    total_batch = 0
    true_pred = 0
    total_loss = 0

    data_loader = tqdm(data_loader)
     
    for batch in data_loader:
        # data bach
        data = {
            'input_ids': batch['input_ids'].to(device),
            'token_type_ids': batch['token_type_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
        }

        #label
        labels = batch['label'].to(device)

        # prediction
        pred = model(data)
        
        # loss
        loss = F.binary_cross_entropy_with_logits(pred, labels)
        
        # optimizer 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # for logging
        pred_y = torch.max(pred,1)[1]
        labels = labels.max(1)[1]
        true_pred += (labels == pred_y).sum().item()

        total_loss += loss.item()

        # batch_size
        total_batch += len(batch)
        
    # logging
    print(f"[!] Epoch{epoch} | loss : {total_loss:.4f} | Batch Acc : {true_pred/total_batch :.2f}")
    log.write(f"[!] Epoch{epoch} | loss : {total_loss:.4f} | Batch Acc : {true_pred/total_batch :.2f}\n")

@torch.no_grad()
def TestModel(model):
    model.eval()
    test_dataset = NSMCDataset("./nsmc/ratings_test.txt", MAX_LEN)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    total_batch = 0
    true_pred = 0
    for batch in test_dataloader:
        # data bach
        data = {
            'input_ids': batch['input_ids'].to(device),
            'token_type_ids': batch['token_type_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
        }

        #label
        labels = batch['label'].to(device)
        
        # prediction
        pred = model(data)
        
        # for logging
        pred_y = torch.max(pred,1)[1]
        true_pred += (labels == pred_y).sum().item()

        total_batch += len(batch)
    
    # logging
    print(f"[!] TEST ACCURACY {true_pred/total_batch :.2f}")
    log.write(f"[!] TEST ACCURACY {true_pred/total_batch :.2f}\n")



model = BertClassifier() 
if torch.cuda.device_count() > 1 and MULTIGPU:
    model = nn.DataParallel(model)
model.to(device)


# train model
if MODE == 'train':
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    train_dataset = NSMCDataset("./nsmc/ratings_train.txt", MAX_LEN)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        TrainOneEpoch(model, optimizer, train_dataloader, device, epoch)
    save_model(model, model_path, EPOCHS)

# load model
elif MODE == 'load':
    load_model(model, model_path, EPOCHS)

# error
else:
    print("[!] Check Configuration 'mode'")
    log.write("[!] Check Configuration 'mode'")



# test model
TestModel(model)

