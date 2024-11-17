from torch import nn
class SE(nn.Module):
    def __init__(self, c1, ratio=16):  #初始化SE模块，c1为通道数，ratio为降维比率
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)   #自适应平均池化层，将特征图空间压缩为1*1
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)   #降维，减少参数量和计算量
        self.relu = nn.ReLU(inplace=True)  #ReLU激活函数，引入非线性
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)  #升维，恢复到原始通道数
        self.sig = nn.Sigmoid()   #Sigmoid激活函数，输出每个通道的重要系数

    def forward(self, x):  #向前传播方法
        b, c, _, _ = x.size()   #获取输入x的批量大小b和通道数c
        y = self.avgpool(x).view(b, c)     #通过自适应平均池化层后，调整形状以匹配全连接层的输入
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)   #通过全连接层计算通道重要性，调整形状以匹配原始特征图的形状
        return x * y.expand_as(x)  #将通道重要性系数应用到原始特征图上，进行特征重新校准

# class SE(nn.Module):
#     def __init__(self, channel, ratio=16,activation=nn.ReLU(inplace=True)):  #初始化SE模块，c1为通道数，ratio为降维比率
#         super(SE, self).__init__()
#         self.avgpool = nn.AdaptiveAvgPool2d(1)   #自适应平均池化层，将特征图空间压缩为1*1
#         self.l1 = nn.Linear(channel, channel // ratio, bias=False)   #降维，减少参数量和计算量
#         self.relu = nn.ReLU(inplace=True)  #ReLU激活函数，引入非线性
#         self.l2 = nn.Linear(channel // ratio, channel, bias=False)  #升维，恢复到原始通道数
#         self.sig = nn.Sigmoid()   #Sigmoid激活函数，输出每个通道的重要系数
#
#     def forward(self, x):  #向前传播方法
#         b, c, _, _ = x.size()   #获取输入x的批量大小b和通道数c
#         y = self.avgpool(x).view(b, c)     #通过自适应平均池化层后，调整形状以匹配全连接层的输入
#         y = self.l1(y)
#         y = self.relu(y)
#         y = self.l2(y)
#         y = self.sig(y)
#         y = y.view(b, c, 1, 1)   #通过全连接层计算通道重要性，调整形状以匹配原始特征图的形状
#         return x * y.expand_as(x)  #将通道重要性系数应用到原始特征图上，进行特征重新校准
#
# class SE(nn.Module):
#      def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):  # 初始化SE模块，c1为通道数，ratio为降维比率
#             super(SE, self).__init__()
#             self.se_1 = SE(channels_in,activation=activation)
#             self.se_2 = SE(channels_in,activation=activation)
#
#      def forward(self, se1,se2):
#             se1 = self.se_1(se1)
#             se2 = self.se_2(se2)
#             out = se1 + se2
#             return out
