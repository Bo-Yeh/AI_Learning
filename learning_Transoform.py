from PIL import Image 
from torchvision import transforms

#自己的小筆記 TransForm 就像是對電腦對圖片加工用妙妙工具箱,跟opencv有一曲同工之妙，但是這個功能較為強大
#將input(img) 透過 TransForm 對圖片進行加工後輸出

img_path = "Image/PetImages/train/Cat/0.jpg"
img = Image.open(img_path)  #定義 img  

#ToTensor 將資料轉為tensor形式
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img) #餵PIL格式去轉換>Tensor

print("原PIL圖片格式:",img)
print("經過tensor工具轉換格式:",tensor_img)

#Normalize 規一化  (更改圖片顏色?!)
tensor_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])   #可自行更改參數做出不一樣的效果 RGB顏色的變化 增加數據集的資料
img_norm = tensor_norm(tensor_img)  #這裡需要注意餵的格式不是PIL,需要使用Tensor的形式
print("tensor:",tensor_img[0][0][0])
print("norm:",img_norm[0][0][0])   #(X*2) - 1

#Resize (調整圖片size)
trans_res = transforms.Resize((512,512)) #設定參數size Weigh higt
img_res = trans_res(img) #餵圖片格式 PIL         
#{注意}這時的img_res 格式依然是 PIL 如果要繼續使用其他功能可能需要轉為Tensor
#Ex:我要先調整圖片大小然後用規一化去增強我的資料就需要 Input(PIL_IMG) -> Resize -> Tensor -> Norma -> ...

#Compose(等比縮放) 
#基本跟Resize概念差不多 輸入須為一個列表且為transform型態,參考:Compose( [Transform參數1],[Transform參數2],... )
trans_res_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_res_2,tensor_trans]) #這邊餵PIL 但把她轉為 tensor了 
img_compose = trans_compose(img)
print(img_compose)

#RandomCrop(隨機裁減) 
trans_randomcrop = transforms.RandomCrop(120) # 填入你要隨機裁減的size 
trans_compose_2 = transforms.Compose([trans_randomcrop,tensor_trans])#餵PIL 一樣轉 tensor形式 把裁減的東西等比縮放
for i in range(5):
    img_crp = trans_compose_2(img)
    print(img_crp) 

#小總結一下 基本transform  (程式版photoshop)
#定義你的工具 = trans.工具(參數)
#對圖片使用你的工具 -> img = 工具(你的圖片)  
#需要檢查一下工具使用時 所需要的{輸入} 類型是PIL or Tensor ?   -> print(typre())
#檢查輸出的類型是甚麼? 
#官方文檔有配置說明文件要多看
#TransBoard 不給我用真的沒差 至少看這些數字還是看了初一些端倪 嗚嗚




