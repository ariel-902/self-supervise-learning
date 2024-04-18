import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from tqdm.auto import tqdm
import random

epoch_num = 300

myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

trainDataPath = './hw1_data_4_students/hw1_data/p2_data/office/train'
testDataPath = './hw1_data_4_students/hw1_data/p2_data/office/val'
# Transforms
train_tfm = transforms.Compose([
    transforms.Resize((128)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])
test_tfm = transforms.Compose([
    transforms.Resize((128)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ImgDataset(Dataset):
    def __init__(self,path,tfm=test_tfm,files = None):
        super(ImgDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label

        return im,label,fname

def fix(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet50(weights = None) 
    model.load_state_dict(torch.load(f'./epoch{epoch_num}sslPretrain.pt'),strict=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 65)
    fix(model) # when used to train scenario E
    model = model.to(device)

    batch_size = 64
    n_epochs = 50
    patience = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_set = ImgDataset(trainDataPath, tfm=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_set = ImgDataset(testDataPath, tfm=test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    stale = 0
    best_acc = 0
    trainLosses = []

    for epoch in range(n_epochs):

        model.train()
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):

            imgs, labels, _= batch
            logits = model(imgs.to(device)) # return float tensor
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            # scheduler.step()

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        trainLosses.append(train_loss)
        # scheduler.step(train_loss)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        model.eval()
        valid_loss = []
        valid_accs = []
        for batch in tqdm(valid_loader):
            imgs, labels,_ = batch
            with torch.no_grad():
                logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # update logs
        if valid_acc > best_acc:
            with open(f"./E_log.txt","a") as F:
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best", file=F)
        else:
            # with open(f"./_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"E_Adam{epoch_num}_best.ckpt") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break


