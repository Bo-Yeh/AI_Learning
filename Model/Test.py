from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import os, torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import optim
from torchsummary import summary
import torch.nn.functional as F
import time

# cuda or cpu ?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 實作一個可以讀取 stanford dog (mini) 的 Pytorch dataset
class SkinDataset(Dataset):
    
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames    # 資料集的所有檔名
        self.labels = labels          # 影像的標籤
        self.transform = transform    # 影像的轉換方式
 
    def __len__(self):
        return len(self.filenames)    # return DataSet 長度
 
    def __getitem__(self, idx):       
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        label = np.array(self.labels[idx])
        label = torch.tensor(label, dtype=torch.long)  # 將標籤轉換為LongTensor類型
        return image, label



normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

# Transformer
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 調整色度、亮度、飽和度、對比度
    transforms.ToTensor(),
    normalize
])
 
test_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


def split_Train_Val_Data(data_dir):
        
    dataset = ImageFolder(data_dir)
    character = [[] for _ in range(len(dataset.classes))]
    
    for x, y in dataset.samples:
        character[y].append(x)
      
    train_inputs, test_inputs = [], []
    train_labels, test_labels = [], []
    
    for i, data in enumerate(character):
        np.random.seed(42)
        np.random.shuffle(data)
            
        num_sample_train = int(len(data) * 0.8)
        num_sample_test = len(data) - num_sample_train
        
        for x in data[:num_sample_train]:
            train_inputs.append(x)
            train_labels.append(i)
            
        for x in data[num_sample_test:]:
            test_inputs.append(x)
            test_labels.append(i)

    train_dataloader = DataLoader(
        SkinDataset(train_inputs, train_labels, train_transformer),
        batch_size=batch_size, shuffle=True
    )
    
    test_dataloader = DataLoader(
        SkinDataset(test_inputs, test_labels, test_transformer),
        batch_size=batch_size, shuffle=False
    )
    
    return train_dataloader, test_dataloader

# 參數設定
batch_size = 16                                # Batch Size
lr = 0.001                                    # Learning Rate
epochs = 150                                       # epoch 次數

data_dir = 'D:/skincaner/test'                        # 資料夾名稱

train_dataloader, test_dataloader = split_Train_Val_Data(data_dir)

C = models.resnet50(pretrained=True).to(device)     # 使用內建的 model   >>>> model = torchvision.models.vgg16(weights='IMAGENET1K_V1')
optimizer_C = optim.Adam(C.parameters (), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False) # 選擇你想用的 optimizer
summary(C, (3, 244, 244))                        # 利用 torchsummary 的 summary package 印出模型資訊，input size: (3 * 224 * 224)
# Loss function
criterion = nn.CrossEntropyLoss()                # 選擇想用的 loss function

loss_epoch_C = []
train_acc, test_acc = [], []
best_acc, best_auc = 0.0, 0.0

if __name__ == '__main__':    
    for epoch in range(epochs):
        start_time = time.time()
        iter = 0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0
        train_loss_C = 0.0

        C.train() # 設定 train 或 eval
        print('epoch: ' + str(epoch + 1) + ' / ' + str(epochs))  
        
        # ---------------------------
        # Training Stage
        # ---------------------------
        for i, (x, label) in enumerate(train_dataloader) :
            x, label = x.to(device), label.to(device)
            optimizer_C.zero_grad()                         # 清空梯度
            train_output = C(x)                             # 將訓練資料輸入至模型進行訓練 (Forward propagation)
            train_loss = criterion(train_output, label)     # 計算 loss
            train_loss.backward()                           # 將 loss 反向傳播
            optimizer_C.step()                              # 更新權重
            
            
            # 計算訓練資料的準確度 (correct_train / total_train)
            _, predicted = torch.max(train_output.data, 1)  # 取出預測的 maximum
            total_train += label.size(0)
            correct_train += (predicted == label).sum()
            train_loss_C += train_loss.item()
            iter += 1
                    
        print('Training epoch: %d / loss_C: %.3f | acc: %.3f' % \
              (epoch + 1, train_loss_C / iter, correct_train / total_train))
        
        # --------------------------
        # Testing Stage
        # --------------------------
        C.eval() # 設定 train 或 eval
        for i, (x, label) in enumerate(test_dataloader) :
            with torch.no_grad():                           # 測試階段不需要求梯度
                x, label = x.to(device), label.to(device)
                test_output = C(x)                          # 將測試資料輸入至模型進行測試
                test_loss = criterion(test_output, label)   # 計算 loss
                
                # 計算測試資料的準確度 (correct_test / total_test)
                _, predicted = torch.max(test_output.data, 1)
                total_test += label.size(0)
                correct_test += (predicted == label).sum()
        
        print('Testing acc: %.3f' % (correct_test / total_test))
                                     
        train_acc.append(100 * (correct_train / total_train).cpu()) # training accuracy
        test_acc.append(100 * (correct_test / total_test).cpu())    # testing accuracy
        loss_epoch_C.append((train_loss_C / iter))            # loss 

        end_time = time.time()
        print('Cost %.3f(secs)' % (end_time - start_time))

fig_dir = './fig/'
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)
    os.environ['KMP_DUPLICATE_LIB_OK']='True'


plt.figure()
plt.plot(list(range(epochs)), loss_epoch_C) # plot your loss
plt.title('Training Loss')
plt.ylabel('loss'), plt.xlabel('epoch')
plt.legend(['loss_C'], loc = 'upper left')
plt.savefig(os.path.join(fig_dir, 'loss.png'))
plt.close() 

plt.figure()
plt.plot(list(range(epochs)), train_acc)    # plot your training accuracy
plt.plot(list(range(epochs)), test_acc)     # plot your testing accuracy
plt.title('Training acc')
plt.ylabel('acc (%)'), plt.xlabel('epoch')
plt.legend(['training acc', 'testing acc'], loc = 'upper left')
plt.savefig(os.path.join(fig_dir, 'acc.png'))
plt.close() 

print('done')