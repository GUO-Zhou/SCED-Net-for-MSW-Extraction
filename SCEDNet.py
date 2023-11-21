import torch
import torch.nn as nn

from CNN_encoder import CNN, DoubleConv
from Swin_Transformer_encoder import SwinTransformer

class ConvTranspose(nn.Module):
    def __init__(self, in_ch, stride=2):
        super().__init__()
        
        self.layer = nn.Sequential(
            DoubleConv(in_ch=in_ch, out_ch=in_ch//2),
            nn.ConvTranspose2d(in_channels=in_ch//2, out_channels=in_ch//4, kernel_size=2, stride=stride, output_padding=0 if stride==2 else 2)
            )
    
    def forward(self, x):
        x = self.layer(x)
        return x

class SCEDNet(nn.Module):
    """Decoder of SCED-Net.

    Parameters:
        mode (str): Mode of decoder. It should be "class" or "segment". Default: "class"
        num_classes (int): Number of classes for classification head. Default: 1000
        img_size (int): Input image size. Default: 224
        patch_size (int): Patch size. Default: 4
        in_ch (int): Number of input channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """
    def __init__(self, mode="class", num_classes=1000, img_size=224, patch_size=4, in_ch=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4, qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()
        self.mode = mode
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.ST_layer = SwinTransformer(img_size=img_size, patch_size=patch_size, in_ch=in_ch, embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, act_layer=act_layer, norm_layer=norm_layer, patch_norm=patch_norm)
        
        self.CNN_layer = CNN(in_ch=in_ch, init_feature=embed_dim // 2)

        # class
        self.norm = norm_layer(self.num_features * 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.class_head = nn.Linear(self.num_features * 2, num_classes)

        # segment
        self.decoder_layer = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ConvTranspose(2 * self.num_features // (2 ** i_layer), stride=2 if i_layer < (self.num_layers - 1) else 4)
            self.decoder_layer.append(layer)
        head_ch = self.num_features // (2 ** (self.num_layers))
        self.seg_head = nn.Sequential(
            DoubleConv(in_ch=head_ch, out_ch=head_ch // 2),
            nn.Conv2d(in_channels=head_ch // 2, out_channels=num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        x_CNN = self.CNN_layer(x)
        x_ST, ST_layer = self.ST_layer(x)
        
        assert self.mode == "class" or "segment", "Mode should be 'class' or 'segment'"
        if self.mode == "class":
            x_CNN = x_CNN.reshape(x_CNN.shape[0], -1, x_CNN.shape[1])
            x = torch.cat([x_ST, x_CNN], dim=2)
            x = self.norm(x)
            x = self.avgpool(x.transpose(1, 2))
            x = torch.flatten(x, 1)
            x = self.class_head(x)
            x = nn.Softmax()(x)
        elif self.mode == "segment":
            x = x_CNN
            for i_layer in range(self.num_layers):
                x = torch.cat([x, ST_layer[-i_layer-1]], dim=1)
                layer = self.decoder_layer[i_layer]
                x = layer(x)
            x = self.seg_head(x)
            x = nn.Sigmoid()(x)
        
        return x

if __name__ == "__main__":
    model = SCEDNet(mode="segment", num_classes=1)
    model.train()
    print(model(torch.rand(1, 3, 224, 224)).shape)