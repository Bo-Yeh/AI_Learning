from torch.utils.data import Dataset, DataLoader
from PIL import Image 


class mydata(Dataset):
    def __init__(self):

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)
    

img_path = "Easy_Modle/PetImages/train/Cat/0.jpg"
img = Image.open(img_path)