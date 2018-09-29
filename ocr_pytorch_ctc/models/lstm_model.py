import torch
from torch import nn

class LSTM_MODEL(torch.nn.Module):
    def __init__(self):
        super(LSTM_MODEL,self).__init__()
        self.conv5_32=nn.Conv2d(1,32,kernel_size=5,padding=2)
        self.conv5_64=nn.Conv2d(1,64,kernel_size=5,padding=2)
        self.relu=nn.ReLU()
        self.conv3_64=nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.conv3_128=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.dropout=nn.Dropout()
        self.batch32=nn.BatchNorm2d(32)
        self.batch64=nn.BatchNorm2d(64)
        self.batch128=nn.BatchNorm2d(128)
        self.embed=nn.Embedding(18,32)
        self.linear_180=nn.Linear(512,180)
        #self.pool=nn.MaxPool2d(kernel_size=2)
        self.linear_11=nn.Linear(1024,11)
        self.lstm=nn.LSTM(input_size=1024,hidden_size=1024,num_layers=2)

        self.pool=nn.MaxPool2d((2, 2))
    def forward(self, image,batch_size=64):


        f=self.conv5_64(image)
        f=self.batch64(f)
        f=self.relu(f)
        f=self.pool(f)

        f=self.conv3_128(f)
        f=self.batch128(f)
        f=self.relu(f)
        f=self.pool(f)
        f=f.contiguous()
        f=f.view(32,-1,64)




        f=f.permute(2,0,1).contiguous()
        f,_=self.lstm(f)
        t,b,h = f.size()
        f=f.view(b*t,1024)
        f=self.linear_11(f)



        f=f.view(t,b,-1)
    #    f=f.permute(1,0,2)
        return f