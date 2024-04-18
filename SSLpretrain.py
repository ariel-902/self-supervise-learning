import torch
from byol_pytorch import BYOL
import torchvision.transforms as transforms
import os
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import models

trainDataPath = './hw1_data_4_students/hw1_data/p2_data/mini/train'
train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

class ImgDataset(Dataset):
    def __init__(self,path,tfm=train_tfm,files = None):
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
        return im

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet = models.resnet50(weights = None)
resnet = resnet.to(device)
learner = BYOL(
    resnet,
    image_size = 128,
    hidden_layer = 'avgpool'
)
batch_size = 64
epoch_num = 300
train_set = ImgDataset(trainDataPath, tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)

# best_loss = 10000
# patience = 5
# stale = 0
for epoch in range(epoch_num):
    loss_list = []
    for batch in tqdm(train_loader):
        images = batch
        loss = learner(images.to(device))
        optimizer.zero_grad()
        loss.backward()
        loss_list.append(loss.item())
        optimizer.step()
        learner.update_moving_average() # update moving average of target encoder
    
    current_loss = sum(loss_list)/len(loss_list)
    print("epoch", epoch+1, "average loss", current_loss)

    if((epoch+1) % 20 == 0):    #save every 50 epochs
        print("saving model from epoch", epoch+1)
        torch.save(resnet.state_dict(), f'./epoch{epoch+1}sslPretrain.pt')


    # if(current_loss < best_loss):
    #     print("saving model from epoch", epoch+1)
    #     # save your improved network
    #     torch.save(resnet.state_dict(), './sslPretrain.pt')
    #     best_loss = current_loss
    #     stale = 0
    # else:
    #     stale += 1
    #     if (stale >= patience):
    #         print(f"No improvment {patience} consecutive epochs, early stopping")
    #         break

