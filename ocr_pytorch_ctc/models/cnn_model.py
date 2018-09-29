
import torch
from torch import nn
class CNN_MODEL(torch.nn.Module):
    def __init__(self):
        super(CNN_MODEL,self).__init__()
        self.conv5_32=nn.Conv2d(1,32,kernel_size=5,padding=2)
        self.conv5_64=nn.Conv2d(32,64,kernel_size=5,padding=2)
        self.relu=nn.ReLU()
        self.conv3_64=nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.dropout=nn.Dropout()
        self.batch32=nn.BatchNorm2d(32)
        self.batch64=nn.BatchNorm2d(64)
        self.batch128=nn.BatchNorm2d(128)
        self.linear_11=nn.Linear(128,11)
        self.linear_180=nn.Linear(1024,180)
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.pool_21 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.conv2_128=nn.Conv2d(64,128,kernel_size=2)
    def forward(self, image,batch_size=4):
         f = self.conv5_32(image)
         f = self.batch32(f)
         f = self.relu(f)
         f = self.pool(f)
         f = self.dropout(f)

         f = self.conv5_64(f)
         f = self.batch64(f)
         f = self.relu(f)
         f = self.pool(f)
         f = self.dropout(f)

         f = self.conv3_64(f)
         f = self.batch64(f)
         f = self.relu(f)
         f = self.pool_21(f)
         f = self.dropout(f)

         f = self.conv3_64(f)
         f = self.batch64(f)
         f = self.relu(f)
         f = self.pool_21(f)
         f = self.dropout(f)

         f = self.conv2_128(f)
         f = self.batch128(f)
         f = self.relu(f)
         f = f.squeeze()
         f = f.permute(2,0,1).contiguous()
         t,b,h=f.size()
         f = f.view(t*b,h)
         f = self.linear_11(f)
         output = f.view(t,b,-1)




         return output