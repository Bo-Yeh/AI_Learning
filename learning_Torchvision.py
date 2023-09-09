import torchvision 
from torchvision import transforms
from PIL import Image

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()   #轉為totensor 樣式
])



train = torchvision.datasets.CIFAR10(root="./torchversion_dateset",train=True,transform= data_transform, download=True) #root=目標路經,Train=True訓練反之為測試,download 選擇是否從網上下載dataset,transform選擇是否需要使用transform處理
test = torchvision.datasets.CIFAR10(root="./torchversion_dateset",train=False,transform= data_transform,download=True) 



print(test[0]) 
# img , memu = test[0]
# print("image:",img)
# print("menu:",memu)
# img.show()