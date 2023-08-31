from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import os  #獲取資料夾下的圖片

#初始化讀取資料
class Mydata(Dataset):
    def __init__(self,memu_dir,lable_dir):
        self.memu_dir = memu_dir
        self.lable_dir = lable_dir
        self.path = os.path.join(self.memu_dir,self.lable_dir) #路徑融合
        self.img_path = os.listdir(self.path) #lable下的所有圖片位置 ex:/train/cat/001.jpg , 002.jpg , 003.jpg ...
    def __getitem__(self, index) :
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.memu_dir,self.lable_dir,img_name) #[第X]張圖片的路徑  ex:/cat/001.jpg
        img = Image.open(img_item_path)
        lable = self.lable_dir
        return img,lable
    def __len__(self): #計算資料長度
        return(len(self.img_path))



# img_path = "Image/PetImages/train_small/Cat/0.jpg"
# img = Image.open(img_path)
# img.show()

#test
memu_dir = "Image/PetImages/train_small"
cat_lable_dir = "Cat"
dog_lable_dir = "dog"
cat_dataset = Mydata(memu_dir,cat_lable_dir)
dog_dataset = Mydata(memu_dir,dog_lable_dir)
train_dataset = cat_dataset + dog_dataset 
# print(len(cat_dataset))
# img,lable = dog_dataset[0]
# img.show()