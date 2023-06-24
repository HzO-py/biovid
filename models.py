from torch.nn.utils import weight_norm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
from torch.nn.modules.loss import _Loss
from torch import log2,exp
import math

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 7)
        #self.linear=nn.Linear(512,1)

    def forward(self, x):
        out = self.features(x)
        fea = out.view(out.size(0), -1)
        #out = F.dropout(fea, p=0.5, training=self.training)
        #out = self.linear(out)
        return fea

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG_regressor(nn.Module):
    def __init__(self):
        super(VGG_regressor, self).__init__()
        self.VGG = VGG('VGG19')
        self.linear=nn.Sequential(nn.Linear(512,256),nn.Linear(256,64),nn.Linear(64,1))

    def forward(self, x):
        fea = self.VGG(x)
        print(fea.size())
        out=self.linear(fea)
        #out = F.dropout(fea, p=0.5, training=self.training)
        #out = self.linear(out)
        return out,fea,None

class Resnet_regressor(nn.Module):
    def __init__(self,modal,is_gradcam=False):
        super(Resnet_regressor, self).__init__()
        pretrained=True if modal=='face' else False
        resnet=torchvision.models.resnet18(pretrained=pretrained)
        if modal=='D':
            resnet.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.resnet=nn.Sequential(*list(resnet.children())[:-1])
        self.linear=nn.Linear(512,1)
        self.fea_grad=None
        self.is_gradcam=is_gradcam

    def gram_func(self,x):
        self.fea_grad=x

    def forward(self, x):
        res=None
        for i in range(len(self.resnet)):
            x=self.resnet[i](x)
            if i==len(self.resnet)-2 and self.is_gradcam:
                x.register_hook(self.gram_func)
                res=x.clone()
        fea = x.view(x.size(0), -1)

        out=self.linear(fea)
        return out,fea,res

class Resnet_classifier(nn.Module):
    def __init__(self,modal,class_num,is_gradcam=False):
        super(Resnet_classifier, self).__init__()
        pretrained=True if modal=='face' else False
        resnet=torchvision.models.resnet18(pretrained=pretrained)
        if modal=='D':
            resnet.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.resnet=nn.Sequential(*list(resnet.children())[:-1])
        self.linear=nn.Linear(512,class_num)
        self.fea_grad=None
        self.is_gradcam=is_gradcam

    def gram_func(self,x):
        self.fea_grad=x

    def forward(self, x):
        res=None
        for i in range(len(self.resnet)):
            x=self.resnet[i](x)
            if i==len(self.resnet)-2 and self.is_gradcam:
                x.register_hook(self.gram_func)
                res=x.clone()
        fea = x.view(x.size(0), -1)

        out=self.linear(fea)
        return out,fea,res

#like huajun
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)   
# import torchsummary
# model = Resnet_regressor('face').cuda()
# torchsummary.summary(model, (3, 224, 224))
# print('parameters_count:',count_parameters(model))
#like huajun

class NoChange(nn.Module):#自定义类 继承nn.Module

    def __init__(self):#初始化函数
        super(NoChange, self).__init__()#继承父类初始化函数

    def forward(self, x):
        return x,x,x

class ClusterCenter(nn.Module):
    def __init__(self,hiddenNum,cluster_num):
        super(ClusterCenter, self).__init__()
        self.fc1 = nn.Linear(hiddenNum, cluster_num, bias = False)

    def forward(self, x):
        centers = list(self.fc1.parameters())
        return x,centers

class Cluster(nn.Module):
    def __init__(self,inputNum,hiddenNum,cluster_num):
        super(Cluster, self).__init__()
        self.cluster_num=cluster_num
        self.fc1 = nn.Linear(inputNum, hiddenNum)
        self.fc2 = nn.Linear(hiddenNum, cluster_num, bias = False)

    def forward(self, x):
        x=self.fc1(x)
        centers = list(self.fc2.parameters())
        return x,centers

class Prototype(nn.Module):#自定义类 继承nn.Module

    def __init__(self,inputNum,hiddenNum,outputNum):#初始化函数
        super(Prototype, self).__init__()#继承父类初始化函数
        self.inputNum=inputNum
        self.hiddenNum=hiddenNum
        self.outputNum=outputNum
        self.fc1 = nn.Linear(inputNum, hiddenNum)
        self.fc2 = nn.Linear(hiddenNum, outputNum, bias = False)

    def forward(self, x):
        out = self.fc1(x)
  
        fc_w1 = list(self.fc1.parameters())
        fc_w2 = list(self.fc2.parameters())

        return out,fc_w1,fc_w2

class Regressor(nn.Module):# 最终的分类器，用于输出预测概率

    def __init__(self,inputNum,hiddenNum):#初始化函数
        super(Regressor, self).__init__()#继承父类初始化函数
        self.fc1 = nn.Linear(inputNum, hiddenNum)
        #self.fc2 = nn.Linear(hiddenNum, hiddenNum, bias = True)
        self.fc3 = nn.Linear(hiddenNum, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.selu(x)
        # x = self.fc2(x)
        # fea = F.selu(x)
        out = self.fc3(x)
        return out

class Classifier(nn.Module):# 最终的分类器，用于输出预测概率

    def __init__(self,inputNum,hiddenNum,outputNum):#初始化函数
        super(Classifier, self).__init__()#继承父类初始化函数
        self.fc1 = nn.Linear(inputNum, hiddenNum)
        self.fc2 = nn.Linear(hiddenNum, outputNum)
        self.num_classes=outputNum

    def forward(self, x):
        x = self.fc1(x)
        x = F.selu(x)
        out = self.fc2(x)
        return out
    
class Classifierxy(nn.Module):# 最终的分类器，用于输出预测概率

    def __init__(self,inputNum1,inputNum2,hiddenNum,outputNum):#初始化函数
        super(Classifierxy, self).__init__()#继承父类初始化函数
        self.fc1 = nn.Linear(inputNum1, hiddenNum)
        self.fc2 = nn.Linear(inputNum2, hiddenNum)
        self.fc3 = nn.Linear(inputNum1+inputNum2, outputNum)
        self.num_classes=outputNum

    def forward(self, x,y):
        x = self.fc1(x)
        x = F.selu(x)
        y = self.fc2(y)
        y = F.selu(y)
        all=torch.cat([x,y],-1)
        out = self.fc3(all)
        return out

class Classifierxplusy(nn.Module):# 最终的分类器，用于输出预测概率

    def __init__(self,inputNum1,inputNum2,hiddenNum,outputNum):#初始化函数
        super(Classifierxplusy, self).__init__()#继承父类初始化函数
        # self.fc1 = nn.Linear(inputNum1, hiddenNum)
        # self.fc2 = nn.Linear(inputNum2, hiddenNum)
        self.fc3 = nn.Linear(inputNum1+inputNum2, hiddenNum)
        self.fc4 = nn.Linear(hiddenNum, outputNum)
        self.num_classes=outputNum

    def forward(self, x,y):
        # x = self.fc1(x)
        # x = F.selu(x)
        # y = self.fc2(y)
        # y = F.selu(y)
        all=torch.cat([x,y],-1)
        out = self.fc3(all)
        out = F.selu(out)
        out = self.fc4(out)
        return out

class Regressor2(nn.Module):# 最终的分类器，用于输出预测概率

    def __init__(self,inputNum,hiddenNum):#初始化函数
        super(Regressor2, self).__init__()#继承父类初始化函数
        self.fc1 = nn.Linear(inputNum, hiddenNum)
        self.fc2 = nn.Linear(hiddenNum, hiddenNum)
        self.fc3 = nn.Linear(hiddenNum, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc2(x)
        x = F.selu(x)
        x = self.fc3(x)
        out=torch.sigmoid(x)
        return x

class Regressor_self_att(nn.Module):# 最终的分类器，用于输出预测概率
    def __init__(self,inputNum,hiddenNum_1, hiddenNum_2,is_droup):#初始化函数
        super(Regressor_self_att, self).__init__()#继承父类初始化函数
        self.is_droup = is_droup

        self.att0 = nn.Linear(inputNum, hiddenNum_1, bias = True)
        self.att1 = nn.Linear(hiddenNum_1, inputNum, bias = False)

        self.fc1 = nn.Linear(inputNum, hiddenNum_1, bias = True)
        if is_droup is not None:
            self.fc1_droupout = nn.Dropout(p=is_droup)
        self.fc2 = nn.Linear(hiddenNum_1, hiddenNum_2, bias = True)
        if is_droup is not None:
            self.fc2_droupout = nn.Dropout(p=is_droup)
        self.fc3 = nn.Linear(hiddenNum_2, 1, bias = False)
    
    def forward(self, x):
        att=self.att0(x)
        att=self.att1(att)
        x=x*att
        x = self.fc1(x)
        x = F.relu(x)
        if self.is_droup:
            x = self.fc1_droupout(x)
        x = self.fc2(x)
        x = F.relu(x)
        if self.is_droup:
            x = self.fc2_droupout(x)
        out = self.fc3(x)
        return out,att

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=28):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer5 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer6 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        
        #self.fc = nn.Linear(512, 28)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        #out = self.layer6(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        xx = out
        #out = self.fc(out)

        return xx


def ResNet18():

    return ResNet(ResidualBlock)

class cnn1d(nn.Module):#自定义类 继承nn.Module

    def __init__(self,outputNum):#初始化函数
        super(cnn1d, self).__init__()#继承父类初始化函数

        self.fc1 = nn.Linear(22336, 256, bias = True)

        self.fc2 = nn.Linear(256, outputNum, bias = True)   
   
        #self.fc3 = nn.Linear(256, 32, bias = True)
 
        #self.fc4 = nn.Linear(128, 32, bias = True)
        self.stride=2
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=self.stride)
        
    def forward(self, x):
        # x = F.tanh(self.fc1(x))
        # x = F.tanh(self.fc2(x))
        # x = F.tanh(self.fc3(x))
        # out = F.tanh(self.fc4(x))
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = self.max_pool1(x)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.max_pool1(x)
        #print(x.shape)
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = self.max_pool1(x)
        #print(x.shape)
        x = x.view(x.size(0),-1)
        #print(x.shape)
   
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        out = x
        return out

class voiceCNN(nn.Module):
    def __init__(self,):
        super(voiceCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=(32,2),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d((4,2), stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=(16,2),
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(8,2),
                stride=1,
            ),
            nn.ReLU(),
        )
        self.input_layer = nn.Linear(864, 2048)
        self.layer_output = nn.Linear(2048, 2)
        self.dropout = nn.Dropout(p=0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        # [b,1,199,13]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_output(x)
        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
        #                          self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, reverse=False):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            if reverse:
                dilation_size = int(pow(2, num_levels-1)/pow(2, i))
            else:
                dilation_size = 2 ** i   # 膨胀系数：1，2，4，8……
            # 确定每一层的输入通道数
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=20, dropout=0.2):
        super(TCN, self).__init__()
        num_channels=[input_size*2,input_size*2]
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        inputs=inputs.transpose(2,1)
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return o,o

class VoiceCNN(nn.Module):
    def __init__(self,):
        super(VoiceCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=(32,2),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d((4,2), stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=(16,2),
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(8,2),
                stride=1,
            ),
            nn.ReLU(),
        )
        self.input_layer = nn.Linear(864, 2048)
        self.layer_output = nn.Linear(2048, 2)
        self.dropout = nn.Dropout(p=0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        # [b,1,199,13]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        x = self.dropout(x)
        fea = self.relu(x)
        x = self.layer_output(fea)
        return x,fea

class BILSTM(nn.Module):
    def __init__(self,channel_num,hidden_size,layer_num,device):
        super(BILSTM, self).__init__()

        self.n_layers=layer_num
        self.n_class=channel_num
        self.n_hidden=hidden_size
        self.device=device
        
        self.lstm = nn.LSTM(input_size=self.n_class, hidden_size=self.n_hidden, bidirectional=True,num_layers=self.n_layers)
        self.W = nn.Parameter(torch.randn([2*self.n_hidden, self.n_class]).type(torch.float32))
        
        self.b = nn.Parameter(torch.randn([self.n_class]).type(torch.float32))

    def forward(self, x):
        batch_size = len(x)
        x = x.transpose(0, 1).contiguous()

        
        init_hidden_state = Variable(torch.zeros(self.n_layers*2, batch_size, self.n_hidden)).to(self.device)
        init_cell_state = Variable(torch.zeros(self.n_layers*2, batch_size, self.n_hidden)).to(self.device)

        outputs, (_, _) = self.lstm(x, (init_hidden_state, init_cell_state))
        outputs = outputs[-1]
        final_output = torch.mm(outputs, self.W) + self.b

        return final_output

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,is_bi = False,dropout=0) -> None:
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers = num_layers, batch_first=True, bidirectional=is_bi, dropout=dropout)
        #self.linear=nn.Linear(hidden_size,1)
        # self.reset_hidden()
    def forward(self, x, hidden=None):
        # self.reset_hidden()
        _,(h, c) = self.lstm(x, hidden)
        #out=self.linear(h[-1])
        return h[-1]

class AU_extractor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AU_extractor, self).__init__()
        self.linear_eye=nn.Sequential(nn.Linear(input_size[0],hidden_size),nn.SELU(),nn.Linear(hidden_size,output_size))
        self.linear_mouth=nn.Sequential(nn.Linear(input_size[1],hidden_size),nn.SELU(),nn.Linear(hidden_size,output_size))
    def forward(self, eye, mouth):
        eye=self.linear_eye(eye)
        mouth=self.linear_mouth(mouth)
        out=torch.cat((eye,mouth),dim=-1)
        return out

def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 
    return loss

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))
        
    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)

class ClusterLoss(_Loss):
    def __init__(self,extractor_num):
        super(ClusterLoss, self).__init__()
        self.pain_center = torch.nn.Parameter(torch.zeros([extractor_num],dtype=torch.float))
        
    def forward(self, pred, target):
        loss=0.0
        for i in range(target.size()[0]):
            tar=float(target[i])
            y=math.log2((tar+1))
            pre=pred[i]
            d=exp(-torch.norm((pre-self.pain_center))**2)
            loss+=-y*log2(d)-(1-y)*log2(1-d)
        loss/=target.size()[0]
        #loss=loss.detach()
        return loss

RNNS = ['LSTM', 'GRU']

class RNNEncoder(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
               bidirectional=True, rnn_type='GRU'):
    super(RNNEncoder, self).__init__()
    self.bidirectional = bidirectional
    assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
    rnn_cell = getattr(nn, rnn_type) # fetch constructor from torch.nn, cleaner than if
    self.rnn = rnn_cell(embedding_dim, hidden_dim, nlayers, 
                        dropout=dropout, bidirectional=bidirectional)

  def forward(self, input, hidden=None):
    return self.rnn(input, hidden)

class SelfAttention(nn.Module):
  def __init__(self, query_dim):
    super(SelfAttention, self).__init__()
    self.scale = 1. / math.sqrt(query_dim)

  def forward(self, query, keys, values):
    # Query = [BxQ]
    # Keys = [TxBxK]
    # Values = [TxBxV]
    # Outputs = a:[TxB], lin_comb:[BxV]

    # Here we assume q_dim == k_dim (dot product attention)

    query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
    keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]

    energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
    energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

    values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
    linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
    
    return energy, linear_combination

class Time_SelfAttention(nn.Module):
  def __init__(self, embedding_dim, hidden_dim):
    super(Time_SelfAttention, self).__init__()
    self.encoder = RNNEncoder(embedding_dim, hidden_dim)
    self.attention = SelfAttention(hidden_dim*2)

    # size = 0
    # for p in self.parameters():
    #   size += p.nelement()
    # print('Total param size: {}'.format(size))


  def forward(self, input):
    x = input.transpose(0, 1).contiguous()
    outputs, hidden = self.encoder(x)
    if isinstance(hidden, tuple): # LSTM
      hidden = hidden[1] # take the cell state

    if self.encoder.bidirectional: # need to concat the last 2 hidden layers
      hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
    else:
      hidden = hidden[-1]

    # max across T?
    # Other options (work worse on a few tests):
    # linear_combination, _ = torch.max(outputs, 0)
    # linear_combination = torch.mean(outputs, 0)
    
    energy, linear_combination = self.attention(hidden, outputs, outputs) 
    return  linear_combination,outputs[-1]

class CrossAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(CrossAttention, self).__init__()
        #定义线性变换函数
        # self.linear_q = nn.Linear(embedding_dim, hidden_dim, bias=False)
        # self.linear_k = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.scale = 1 / math.sqrt(embedding_dim)
        #楚杰的窗口
        n=300
        self.mask=torch.ones((n,n),dtype=torch.float32).cuda()
        inf=-1e9
        for i in range(n):
            for j in range(n):
                if abs(i-j)>=2:
                    self.mask[i][j]=inf

    def forward(self, query,keys,values):

        energy = torch.bmm(query, keys.transpose(1, 2))  # [B*T*Q]*[B*K*T]->[B*T*T]

        #楚杰的窗口
        # n=energy.size()[-1]
        # mask=self.mask[:n,:n].contiguous()
        # energy*=mask
        #归一化获得attention的相关系数
        energy = F.softmax(energy.mul_(self.scale), dim=-1)  

        #attention系数和v相乘，获得最终的得分
        linear_combination = torch.bmm(energy, values)  # [B*T*T]*[B*T*V]->[B*T*V]

        return energy,linear_combination

class Face2Voice_Atention(nn.Module):
    def __init__(self, embedding_dim):
        super(Face2Voice_Atention, self).__init__()
        self.fc=nn.Linear(embedding_dim,1)
        self.dim=embedding_dim

    def forward(self, input,query):
        batch_dim=query.size()[0]
        time_dim=query.size()[1]
        query=query.view(-1,self.dim)
        energy=self.fc(query).squeeze(-1)
        #energy=F.softmax(energy,dim=-1)
        energy=torch.diag(energy).view(batch_dim,time_dim,time_dim)
        output=torch.bmm(energy,input)
        return energy,output

class Voice_Time_CrossAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Voice_Time_CrossAttention, self).__init__()
        self.encoder = RNNEncoder(embedding_dim, hidden_dim)
        self.cross_attention = Face2Voice_Atention(embedding_dim)
        self.self_attention = SelfAttention(hidden_dim*2)
        # size = 0
        # for p in self.parameters():
        #   size += p.nelement()
        # print('Total param size: {}'.format(size))


    def forward(self, input,query):
        energy,x = self.cross_attention(input=input,query=query) 
        x = x.transpose(0, 1).contiguous()
        outputs, hidden = self.encoder(x)

        if isinstance(hidden, tuple): # LSTM
            hidden = hidden[1] # take the cell state

        if self.encoder.bidirectional: # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        # max across T?
        # Other options (work worse on a few tests):
        # linear_combination, _ = torch.max(outputs, 0)
        # linear_combination = torch.mean(outputs, 0)
        
        _, linear_combination = self.self_attention(hidden, outputs, outputs) 
        return  linear_combination,energy