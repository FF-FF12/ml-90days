import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
#先写encoder验证shape变化
class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding =1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace= True),
            nn.Conv2d(out_ch,out_ch,3,padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace= True),
        )
    def forward(self,x):
        return self.block(x)

x= torch.randn(1,1,224,224)
d1 = DoubleConv(1,64)
y = d1(x)


#加下采样
enc = nn.Sequential(
    DoubleConv(1,64),
    nn.MaxPool2d(2),
    DoubleConv(64,128),
    nn.MaxPool2d(2),
    DoubleConv(128,256),
)
y = enc(x)
print(y.shape)
class TinyUnet(nn.Module):
    def __init__(self,in_ch=1,num_classes=1):
        super().__init__()
        self.e1 = DoubleConv(in_ch,64)
        self.e2 = DoubleConv(64,128)
        self.e3 = DoubleConv(128,256)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(256,128,2,stride = 2)
        self.d3 = DoubleConv(128+128,128)
        self.up2 = nn.ConvTranspose2d(128,64,2,stride = 2)
        self.d2 = DoubleConv(64+64,64)
        self.out = nn.Conv2d(64,num_classes,1)

    def forward(self,x):
        h,w = x.shape[2],x.shape[3]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x,(0,pad_w,0,pad_h))
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))

        u3 = self.up3(s3)
        u3 = torch.cat([u3,s2],dim = 1)
        u3 = self.d3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2,s1],dim =1)
        u2 = self.d2(u2)
        out = self.out(u2)[:,:,:h,:w]
        return OrderedDict([("out",out)])
if __name__ == "__name__":
    model = TinyUnet(in_ch =3, num_classes =2)
    out = model(torch.randn(1,3,224,224))
    print("输出：",out["out"].shape)