#TensorBoard 練習
from torch.utils.tensorboard import SummaryWriter

#資料存儲
writer = SummaryWriter("logs")

for i in range(100):
    writer.add_scalar("y=x",i,i)  #Globalstep = x , scalar = y (其他設置詳閱用法)


writer.close()