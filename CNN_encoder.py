import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, act_layer=nn.ReLU):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        if act_layer == nn.ReLU:
            self.act = act_layer(inplace=True)
        else:
            self.act = act_layer()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class CNN(nn.Module):
    """CNN
        A CNN branch of SCED-Net.
    Parameters:
        in_ch (int): Number of input image channels. Default: 3
        init_features (int): Number of features in first block. Default: 48
        num_layers (int): Number of blocks in CNN branch. Default: 5
    """
    def __init__(self, in_ch=3, init_feature=48, num_layers=5):
        super().__init__()

        in_list = [init_feature*(2**i) for i in range(num_layers-1)]
        in_list.insert(0, in_ch)
        out_list = [init_feature*(2**(i)) for i in range(num_layers)]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(num_layers):
            layer = nn.Sequential(
                DoubleConv(in_ch=in_list[i_layer], out_ch=out_list[i_layer]),
                nn.MaxPool2d(2)
            )
            self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x