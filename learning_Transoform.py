from PIL import Image 
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
img_path = "Image/PetImages/train/Cat/0.jpg"
img = Image.open(img_path)  #定義 img 

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

print("原PIL圖片格式:",img)
print("經過tensor工具轉換格式:",tensor_img)

writer = SummaryWriter("logs")

writer.add_image("tensor",tensor_img)
writer.close()