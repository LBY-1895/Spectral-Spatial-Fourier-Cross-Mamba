import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 
import h5py
import os
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from timm.models.layers import trunc_normal_
import kornia
from einops import rearrange, repeat

def up_conv_sig(in_channel, out_channel,kernel_size=3, stride=1, padding=1,scale_factor=2):
    return nn.Sequential(
                nn.Upsample(scale_factor=scale_factor),  # add Upsample
                nn.Conv2d(in_channel,out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Sigmoid(),
    )

def conv_bn_relu(in_channel, out_channel,kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),  # todo: paddint
        nn.BatchNorm2d(out_channel, momentum=0.9, eps=0.001),  # note 默认可以修改
        nn.ReLU()
    )

def conv_bn_relu_max(in_channel, out_channel,kernel_size=3, stride=1, padding=1,max_kernel=2):
    return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(max_kernel),
        )

# 光谱特征提取模块
# class SpectralConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(SpectralConv1D, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         # x shape: [batch, channels, height, width]
#         batch, ch, h, w = x.size()
        
#         # 重塑为光谱序列: [B, Ch, H, W] 变为 [B, H, W, Ch]
#         x_spectral = x.permute(0, 2, 3, 1).contiguous().view(-1, ch)  # [batch*h*w, ch]
        
#         # 应用1D卷积
#         #print("x_spectral:", x_spectral.shape)
#         x_spectral = self.relu(self.conv1(x_spectral))
#         x_spectral = self.relu(self.bn(self.conv2(x_spectral)))
        
#         # 全局光谱池化
#         spectral_feat = torch.mean(x_spectral, dim=2)  # [batch*h*w, out_channels]
        
#         # 恢复空间结构
#         spectral_feat = spectral_feat.view(batch, h, w, -1).permute(0, 3, 1, 2)
#         return spectral_feat

class EnhancedSpectralExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1D = nn.Conv1d(
            in_channels=2,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False
        )
        self.ln1 = nn.LayerNorm(71)
        self.relu = nn.ReLU(inplace=True)

        self.conv1D2 = nn.Conv1d(
            in_channels=16,
            out_channels=28,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False
        )
        self.ln2 = nn.LayerNorm(35)
        self.conv1D3 = nn.Conv1d(
            in_channels=28,
            out_channels=28,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False
        )
        self.ln3 = nn.LayerNorm(17)
        self.conv1D4 = nn.Conv1d(
            in_channels=28,
            out_channels=28,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False
        )
        self.ln4 = nn.LayerNorm(8)

        self.fftconv1 = nn.Conv1d(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False
        )
        self.fftconv2 = nn.Conv1d(
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False
        )
        self.fftconv3 = nn.Conv1d(
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False
        )
        self.fftconv4 = nn.Conv1d(
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False
        )

    def forward(self, x):
        batch, ch, h, w = x.size()
        
        center_row = x[:,:, 3:5, :]  # 64,144,2,8
        center_col = x[:,:, :, 3:5]  # 64,144,8,2

        center_row_fft = center_row.cpu().numpy()
        # 对每个像素的光谱通道进行傅里叶变换
        center_row_fft = np.fft.fft(center_row_fft, axis=1)  # 沿光谱通道轴进行变换
        # 取幅度谱（绝对值）
        center_row_fft = np.abs(center_row_fft)
        center_col_fft = center_col.cpu().numpy()
        center_col_fft = np.fft.fft(center_col_fft, axis=1)  # 沿光谱通道轴进行变换
        center_col_fft = np.abs(center_col_fft)
        # print("center_row:", center_row.shape)
        center_row_fft = torch.from_numpy(center_row_fft).float()
        center_row_fft = center_row_fft.to('cuda:0')  # 假设使用 GPU
        center_col_fft = torch.from_numpy(center_col_fft).float()
        center_col_fft = center_col_fft.to('cuda:0')  # 假设使用 GPU

        center_row_fft = center_row_fft.permute(0, 2, 3, 1).contiguous().view(batch*w, 2, -1)  # (B,144,H,W) -> (B,H,W,144)
        center_col_fft = center_col_fft.permute(0, 2, 3, 1).contiguous().view(batch*h, 2, -1)  # (B,144,H,W) -> (B,H,W,144)
        
        center_row = center_row.permute(0, 2, 3, 1).contiguous().view(batch*w, 2, -1)  # (B,144,H,W) -> (B,H,W,144)
        center_col = center_col.permute(0, 2, 3, 1).contiguous().view(batch*h, 2, -1)  # (B,144,H,W) -> (B,H,W,144)
        # print("center_row:", center_row.shape)
        # print("center_col:", center_col.shape)
        center_row = self.relu(self.ln1(self.conv1D(center_row)))
        center_col = self.relu(self.ln1(self.conv1D(center_col)))
        center_row_fft = self.relu(self.ln1(self.fftconv1(center_row_fft)))
        center_col_fft = self.relu(self.ln1(self.fftconv1(center_col_fft)))
        # print("center_row1:", center_row.shape)
        # print("center_col:", center_col.shape)
        center_row = self.relu(self.ln2(self.conv1D2(center_row)))
        center_col = self.relu(self.ln2(self.conv1D2(center_col)))
        center_row = self.relu(self.ln3(self.conv1D3(center_row)))
        center_col = self.relu(self.ln3(self.conv1D3(center_col)))
        center_row = self.relu(self.ln4(self.conv1D4(center_row)))
        center_col = self.relu(self.ln4(self.conv1D4(center_col)))
        
        center_row_fft = self.relu(self.ln2(self.fftconv2(center_row_fft)))
        center_col_fft = self.relu(self.ln2(self.fftconv2(center_col_fft)))
        center_row_fft = self.relu(self.ln3(self.fftconv3(center_row_fft)))
        center_col_fft = self.relu(self.ln3(self.fftconv3(center_col_fft)))
        center_row_fft = self.relu(self.ln4(self.fftconv4(center_row_fft)))
        center_col_fft = self.relu(self.ln4(self.fftconv4(center_col_fft)))
        # print("center_row2:", center_row.shape)
        center_row = center_row.view(batch, w, -1, 8)
        center_col = center_col.view(batch, h, -1, 8)
        center_row = center_row.permute(0, 2, 1, 3)
        center_col = center_col.permute(0, 2, 1, 3)

        center_row_fft = center_row_fft.view(batch, w, -1, 8)
        center_col_fft = center_col_fft.view(batch, h, -1, 8)
        center_row_fft = center_row_fft.permute(0, 2, 1, 3)
        center_col_fft = center_col_fft.permute(0, 2, 1, 3)
        # print("center_row:", center_row.shape)

        fused = torch.cat([center_row, center_col, center_row_fft, center_col_fft], dim=1)
        # print("fused:", fused.shape)
        return fused

class SpectralExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(144, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.output_size =  (8, 8)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(self.output_size)

        self.conv22 = nn.Conv1d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm1d(out_channels//2)
        

    def forward(self, x, x2, x3):
        batch, ch, h, w = x.size()
        
        # 提取光谱特征
        x_spectral = x.permute(0, 2, 3, 1).contiguous().view(-1, ch).unsqueeze(2)
        spectral_feat = self.relu(self.conv1(x_spectral))
        spectral_feat = self.relu(self.bn(self.conv2(spectral_feat)))
        spectral_feat = torch.mean(spectral_feat, dim=2)
        spectral_feat = spectral_feat.view(batch, h, w, -1).permute(0, 3, 1, 2)

        x_spectral2 = x2.permute(0, 2, 3, 1).contiguous().view(-1, ch).unsqueeze(2)
        spectral_feat2 = self.relu(self.conv1(x_spectral2))
        spectral_feat2 = self.relu(self.bn22(self.conv22(spectral_feat2)))
        spectral_feat2 = torch.mean(spectral_feat2, dim=2)
        spectral_feat2 = spectral_feat2.view(batch, 2*h, 2*w, -1).permute(0, 3, 1, 2)
        spectral_feat2 = self.adaptive_pool(spectral_feat2)

        x_spectral3 = x3.permute(0, 2, 3, 1).contiguous().view(-1, ch).unsqueeze(2)
        spectral_feat3 = self.relu(self.conv1(x_spectral3))
        spectral_feat3 = self.relu(self.bn22(self.conv22(spectral_feat3)))
        spectral_feat3 = torch.mean(spectral_feat3, dim=2)
        spectral_feat3 = spectral_feat3.view(batch, 3*h, 3*w, -1).permute(0, 3, 1, 2)
        spectral_feat3 = self.adaptive_pool(spectral_feat3)
        # print("spectral_feat:", spectral_feat.shape)

        spectral_feat_fusion = torch.cat([spectral_feat, spectral_feat2, spectral_feat3], dim=1)
        # print("spectral_feat_fusion:", spectral_feat_fusion.shape)
        return spectral_feat


# class SpectralConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(SpectralConv1D, self).__init__()
#         self.conv1 = nn.Conv1d(1, out_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         batch, ch, h, w = x.size()
        
#         # 保存原始空间特征
#         spatial_feat = x
        
#         # 提取光谱特征
#         x_spectral = x.permute(0, 2, 3, 1).contiguous().view(-1, ch).unsqueeze(1)
#         spectral_feat = self.relu(self.conv1(x_spectral))
#         spectral_feat = self.relu(self.bn(self.conv2(spectral_feat)))
#         spectral_feat = torch.mean(spectral_feat, dim=2)
#         spectral_feat = spectral_feat.view(batch, h, w, -1).permute(0, 3, 1, 2)
        
#         # 融合空间和光谱特征 (通道维度拼接)
#         combined_feat = torch.cat([spatial_feat, spectral_feat], dim=1)
#         return combined_feat

class cnn_encoder_A_and_B(nn.Module):
    def __init__(self, l1, l2):
        super(cnn_encoder_A_and_B, self).__init__()
        
        # 初始卷积层
        self.conv1 = conv_bn_relu(l1,64, 3, 1, 1)
        self.conv2 = conv_bn_relu(l2, 16, 3, 1, 1)
        
        # 多尺度处理层
        self.conv1_1 = conv_bn_relu_max(64, 128, 3, 1, 1, 1)
        self.conv2_1 = conv_bn_relu_max(16, 64, 3, 1, 1, 1)
        self.conv1_2 = conv_bn_relu_max(64, 128, 3, 1, 1, 2)
        self.conv2_2 = conv_bn_relu_max(16, 64, 3, 1, 1, 2)
        self.conv1_3 = conv_bn_relu_max(64, 128, 3, 1, 1, 3)
        self.conv2_3 = conv_bn_relu_max(16, 64, 3, 1, 1, 3)
        
        # # 光谱特征提取模块（输入64通道，输出32通道光谱特征）
        # self.spectral_conv1_1 = SpectralConv1D(64, 32)
        # self.spectral_conv2_1 = SpectralConv1D(64, 32)
        # self.spectral_conv1_2 = SpectralConv1D(64, 32)
        # self.spectral_conv2_2 = SpectralConv1D(64, 32)
        # self.spectral_conv1_3 = SpectralConv1D(64, 32)
        # self.spectral_conv2_3 = SpectralConv1D(64, 32)
        
        # 加权求和后的融合层（输入通道数变为64+32=96）
        self.post_conv1 = conv_bn_relu(128, 64, 1, 1, 0)
        self.post_conv2 = conv_bn_relu(64, 64, 1, 1, 0)

        # 可学习的权重参数
        self.xishu1 = torch.nn.Parameter(torch.Tensor([0.33]))
        self.xishu2 = torch.nn.Parameter(torch.Tensor([0.33]))
        self.xishu3 = torch.nn.Parameter(torch.Tensor([0.33]))

    def forward(self, x11, x21, x12, x22, x13, x23):
        # 初始特征提取
        x11 = self.conv1(x11)
        x21 = self.conv2(x21)
        x12 = self.conv1(x12)
        x22 = self.conv2(x22)
        x13 = self.conv1(x13)
        x23 = self.conv2(x23)

        # 多尺度处理
        x1_1 = self.conv1_1(x11)
        x1_2 = self.conv1_2(x12)
        x1_3 = self.conv1_3(x13)
        
        x2_1 = self.conv2_1(x21)
        x2_2 = self.conv2_2(x22)
        x2_3 = self.conv2_3(x23)
        
        # 在每个尺度上提取光谱特征（保留空间特征）
        # x1_1_combined = self.spectral_conv1_1(x1_1)  # [bs, 64+32, H, W]
        # x1_2_combined = self.spectral_conv1_2(x1_2)
        # x1_3_combined = self.spectral_conv1_3(x1_3)
        
        # x2_1_combined = self.spectral_conv2_1(x2_1)
        # x2_2_combined = self.spectral_conv2_2(x2_2)
        # x2_3_combined = self.spectral_conv2_3(x2_3)
        
        # 加权求和（对融合后的特征）
        x_add1 = x1_1 * self.xishu1 + x1_2 * self.xishu2 + x1_3 * self.xishu3
        x_add2 = x2_1 * self.xishu1 + x2_2 * self.xishu2 + x2_3 * self.xishu3
        
        # 后处理融合层（减少通道数）
        x_add1 = self.post_conv1(x_add1)  # [bs, 64, H, W]
        x_add2 = self.post_conv2(x_add2)

        return x_add1, x_add2

#################2025.07.09###################
# class cnn_encoder_A_and_B(nn.Module):
#     def __init__(self, l1, l2):
#         super(cnn_encoder_A_and_B, self).__init__()

#         self.conv1 = conv_bn_relu(l1, 12, 3, 1, 1)
# ###
#         self.conv2 = conv_bn_relu(l2, 12, 3, 1, 1)
#         self.conv1_1 = conv_bn_relu_max(12, 64, 3, 1, 1,1)
#         self.conv2_1 = conv_bn_relu_max(12, 64, 3, 1, 1,1)
#         self.conv1_2 = conv_bn_relu_max(12, 64, 3, 1, 1,2)
#         self.conv2_2 = conv_bn_relu_max(12, 64, 3, 1, 1,2)
#         self.conv1_3 = conv_bn_relu_max(12, 64, 3, 1, 1,3)
#         self.conv2_3 = conv_bn_relu_max(12, 64, 3, 1, 1,3)

#         self.xishu1 = torch.nn.Parameter(torch.Tensor([0.33]))  # lamda
#         self.xishu2 = torch.nn.Parameter(torch.Tensor([0.33]))  # 1 - lamda
#         self.xishu3 = torch.nn.Parameter(torch.Tensor([0.33]))  # 1 - lamda


#     def forward(self, x11, x21, x12, x22, x13, x23):

#         x11 = self.conv1(x11) #64,32,8,8
#         x21 = self.conv2(x21) #64,32,8,8
#         x12 = self.conv1(x12) #64,32,16,16
#         x22 = self.conv2(x22) #64,32,16,16
#         x13 = self.conv1(x13) #64,32,24,24
#         x23 = self.conv2(x23) #64,32,24,24

#         x1_1 = self.conv1_1(x11) #64,64,8,8
#         x1_2 = self.conv1_2(x12) #64,64,8,8
#         x1_3 = self.conv1_3(x13) #64,64,8,8


#         x_add1 = x1_1 * self.xishu1 + x1_2 * self.xishu2 + x1_3 * self.xishu3
        
#         x2_1 = self.conv2_1(x21)
#         x2_2 = self.conv2_2(x22)
#         x2_3 = self.conv2_3(x23)

#         x_add2 = x2_1 * self.xishu1 + x2_2 * self.xishu2 + x2_3 * self.xishu3
#         # print('The start feature:','w0:',self.xishu1,'w1:',self.xishu2,'w2:',self.xishu3)

#         return x_add1, x_add2

class SimAM(nn.Module):
        def __init__(self):
            super(SimAM, self).__init__()
            self.e_lambda = nn.Parameter(torch.tensor(0.1))
        # X: input feature [N, C, H, W]
        # lambda: coefficient λ in Eqn (5)
        def forward (self, X, lambda_SimAM=0.1):
            #print("simAM输入X的形状:", X.shape)  # 应该是 [batch_size, channels, height, width]
            # spatial size
            n = X.shape[2] * X.shape[3] - 1
            # square of (t - u)
            d = (X - X.mean(dim=[2,3],  keepdim=True)).pow(2) #keepdim=True保持维度不变，有问题可以去掉
            # d.sum() / n is channel variance
            v = d.sum(dim=[2,3],  keepdim=True) / n
            # E_inv groups all importance of X
            #E_inv = d / (4 * (v + lambda_SimAM)) + 0.5
            E_inv = d / (4 * (v + self.e_lambda)) + 0.5
            return F.relu(X * F.sigmoid(E_inv))
            #return X * F.sigmoid(E_inv)


class cnnModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cnnModule, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        #self.attention3 = SimAM()
        #self.attention5 = SimAM()
        #self.attention7 = SimAM()
        # self.conv_fus = nn.Conv2d(in_channels, out_channels*3, kernel_size=1)

    def forward(self, x):
        # 残差分支
        residual = x
        # print("cnnModule输入x的形状:", x.shape)  # 应该是 [batch_size, channels, height, width]

        f3 = self.relu(self.bn(self.conv3x3(x)))
        f5 = self.relu(self.bn(self.conv5x5(x)))
        f7 = self.relu(self.bn(self.conv7x7(x)))

        #f3 = self.attention3(f3)
        #f5 = self.attention5(f5)
        #f7 = self.attention7(f7)

        f_max = torch.max(torch.max(f3, f5), f7)
        f_sum = f3 + f5 + f7
        f_avg = (f3 + f5 + f7) / 3
        f_fused = torch.cat([f_max, f_sum, f_avg], dim=1)
        # residual = self.conv_fus(residual)
        # residual = torch.cat([residual] * 3, dim=1)  # 将残差分支重复3次以匹配主分支的输出通道数
        # f_fused = f_fused + residual
        # print("图像cnnModule输出f_fused的形状:", f_fused.shape)  # 应该是 [batch_size, channels, height, width]
        return f_fused
        
# class cnnModule1D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(cnnModule1D, self).__init__()
#         self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), padding=(0,1))
#         self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,5), padding=(0,2))
#         self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,7), padding=(0,3))
#         self.bn = nn.BatchNorm2d(out_channels)

#         self.cnnAdjustChannel = nn.Conv2d(out_channels*3, in_channels, kernel_size=1)
#         # self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
#         # self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
#         # self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
#         # self.bn = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         #self.attention3 = SimAM()
#         #self.attention5 = SimAM()
#         #self.attention7 = SimAM()
#         # self.conv_fus = nn.Conv2d(in_channels, out_channels*3, kernel_size=1)

#     def forward(self, x):
#         # 残差分支
#         residual = x
#         # print("cnnModule1D输入x的形状:", x.shape)  # 应该是 [batch_size, channels, height, width]
#         # print("cnnModule输入x的形状:", x.shape)  # 应该是 [batch_size, channels, height, width]

#         f3 = self.relu(self.bn(self.conv3(x)))
#         f5 = self.relu(self.bn(self.conv5(x)))
#         f7 = self.relu(self.bn(self.conv7(x)))

#         #f3 = self.attention3(f3)
#         #f5 = self.attention5(f5)
#         #f7 = self.attention7(f7)

#         f_max = torch.max(torch.max(f3, f5), f7)
#         f_sum = f3 + f5 + f7
#         f_avg = (f3 + f5 + f7) / 3
#         f_fused = torch.cat([f_max, f_sum, f_avg], dim=1)
#         f_fused = self.cnnAdjustChannel(f_fused)
#         return f_fused

# class weight_add(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(weight_add, self).__init__()
#         self.x1 = torch.nn.Parameter(torch.Tensor([0.5]))
#         self.x2 = torch.nn.Parameter(torch.Tensor([0.5]))

#         self.conv33 = nn.Conv2d(in_channels*2, out_channels, kernel_size=3, padding=1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#     def forward(self, x1, x2):
#         # add_result = x1 * self.x1 + x2 * self.x2
#         add_result = torch.cat([x1, x2], dim=1)  # 形状变为 [batch, 2*out_channels, h, w]
#         add_conv = self.relu(self.bn(self.conv33(add_result)))
#         return add_conv
class weight_add(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(weight_add, self).__init__()
        self.x1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.x2 = torch.nn.Parameter(torch.Tensor([0.5]))

        self.conv33 = nn.Conv2d(in_channels*2, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x1, x2):
        # add_result = x1 * self.x1 + x2 * self.x2
        add_result = torch.cat([x1, x2], dim=1)  # 形状变为 [batch, 2*out_channels, h, w]
        add_conv = self.relu(self.bn(self.conv33(add_result)))
        # add_result = add_conv * self.x1 + x3 * self.x2
        return add_conv


###################################################
# class ConvToSequence(nn.Module):
#     def __init__(self, embed_dim):
#         """
#         将卷积输出转换为序列输入（ViT风格）
#         Args:
#             embed_dim: 每个patch的特征维度（通道数）
#         """
#         super(ConvToSequence, self).__init__()
#         self.embed_dim = embed_dim
#         # 初始化可学习的class token（不依赖空间尺寸）
#         self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         # 位置编码将在第一次前向传播时根据输入尺寸初始化
#         self.position_embeddings = None

#     def _init_position_embeddings(self, num_patches, device):
#         """
#         动态初始化位置编码
#         Args:
#             num_patches: 总patch数量
#             device: 设备
#         """
#         if self.position_embeddings is None:
#             self.position_embeddings = nn.Parameter(
#                 torch.randn(1, num_patches, self.embed_dim, device=device)
#             )

#     def forward(self, x):
#         """
#         Args:
#             x: 卷积输出，形状为 (B, C, H, W) 例如 [64, 192, 8, 8]
#         Returns:
#             序列输入，形状为 (B, num_patches + 1, C) 例如 [64, 65, 192]
#         """
#         B, C, H, W = x.shape
#         assert C == self.embed_dim, "通道数需等于embed_dim"
        
#         # 展平空间维度 (H, W) -> (H*W)
#         x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
#         num_patches = H * W
        
#         # 动态初始化位置编码
#         self._init_position_embeddings(num_patches, x.device)
        
#         # 添加位置编码
#         x = x + self.position_embeddings
        
#         # 添加class token
#         class_token = self.class_token.expand(B, -1, -1)  # (B, 1, C)
#         sequence = torch.cat((class_token, x), dim=1)  # (B, num_patches+1, C)
        
#         return sequence
###################################################
class ConvToSequence(nn.Module):
    def __init__(self, patch_size, embed_dim):
        """
        将卷积输出转换为序列输入
        Args:
            patch_size: 小块的边长（宽度和高度）
            embed_dim: 每个小块的特征维度（通道数）
        """
        super(ConvToSequence, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = patch_size * patch_size  # 每个小块内部的位置数量
        self.position_embeddings = nn.Parameter(torch.randn(self.num_patches, embed_dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        """
        将卷积输出转换为序列输入
        Args:
            x: 卷积输出，形状为 (B, C, H, W)[64 192 8 8]/[64 96 8 8]
        Returns:
            序列输入，形状为 (B, L+1, D)[64 65 192]/[64 65 96]
        """
        B, C, H, W = x.shape
        assert H == self.patch_size and W == self.patch_size, "输入的小块大小必须与 patch_size 一致"
        x = x.view(B, C, -1).transpose(1, 2)  # 转换为 (B, L, C)
        x = x + self.position_embeddings  # 添加位置嵌入
        class_token = self.class_token.expand(B, -1, -1)  # 扩展类别标记
        sequence = torch.cat((class_token, x), dim=1)  # 在序列开头添加类别标记
        return sequence


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.scale = 2 * math.pi

    def forward(self, x):
        b, _, h, w = x.size()
        
        # 创建位置坐标网格
        y_embed = torch.arange(h, dtype=torch.float32, device=x.device).cumsum(0)
        x_embed = torch.arange(w, dtype=torch.float32, device=x.device).cumsum(0)
        
        # 归一化
        y_embed = y_embed / y_embed[-1] * self.scale
        x_embed = x_embed / x_embed[-1] * self.scale
        
        # 生成正弦位置编码
        pe = torch.zeros(b, self.num_pos_feats * 2, h, w, device=x.device)
        
        # 为每个位置生成正弦和余弦编码
        for i in range(self.num_pos_feats // 2):
            pe[:, 2 * i, :, :] = torch.sin(x_embed * torch.exp(torch.tensor(-math.log(10000.0) / self.num_pos_feats * 2 * i, device=x.device)))
            pe[:, 2 * i + 1, :, :] = torch.cos(y_embed * torch.exp(torch.tensor(-math.log(10000.0) / self.num_pos_feats * 2 * i, device=x.device)))
        
        return pe

class ConvToSequenceSin(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.pos_enc = PositionEmbeddingSine(embed_dim // 2)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.proj(x)  # [b, d, h, w]
        pos = self.pos_enc(x)  # 空间位置编码
        x = (x + pos).flatten(2).transpose(1, 2)  # [b, h*w, d]
        return x


############################0611###############################

class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

class OneDConvNet(nn.Module):
    """
    1D卷积神经网络
    """
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1):
        """
        初始化方法
        Args:
            input_channels: 输入通道数
            output_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
        """
        super(OneDConvNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入数据，形状为 (B, C, L)
        Returns:
            输出数据，形状为 (B, output_channels, L')
        """
        x = self.conv1(x)  # 卷积操作
        x = self.bn1(x)    # 批量归一化
        x = self.relu(x)   # 激活函数
        x = self.pool(x)   # 池化操作
        return x


class MultiModalAttentionFusion(nn.Module):
   def __init__(self, feature_dim, hidden_dim):
       super(MultiModalAttentionFusion, self).__init__()
       self.feature_dim = feature_dim
       self.hidden_dim = hidden_dim
       
       # 调整特征维度的线性层
       self.adjust_dim = nn.Linear(hidden_dim * 3, hidden_dim)
       self.norm_adjust = nn.LayerNorm(hidden_dim)  # 新增：调整维度后的归一化

       # 线性变换层 + LayerNorm（每个模态独立）
       self.fc1 = nn.Linear(feature_dim, hidden_dim)
       self.norm1 = nn.LayerNorm(hidden_dim)  # 新增：hsi_seq 归一化
       self.fc2 = nn.Linear(feature_dim, hidden_dim)
       self.norm2 = nn.LayerNorm(hidden_dim)  # 新增：lidar_seq 归一化
       self.fc3 = nn.Linear(feature_dim, hidden_dim)
       self.norm3 = nn.LayerNorm(hidden_dim)  # 新增：hsi_mamba 归一化

       # 注意力权重层
       self.attention = nn.Linear(hidden_dim, 3)  # 输出3个权重，对应3个输入模态

       # 输出层
       self.fc_out = nn.Linear(hidden_dim, feature_dim)
       self.norm_out = nn.LayerNorm(feature_dim)  # 新增：输出层归一化

       self.dropout = nn.Dropout(0.2)  # 新增：Dropout laye

   def forward(self, hsi_seq, lidar_seq, hsi_mamba):
       # 输入特征的形状为 [batch_size, seq_length, feature_dim]
       batch_size, seq_length, _ = hsi_seq.size()

       # 将输入特征映射到隐藏空间
       hsi_hidden = F.relu(self.fc1(hsi_seq))  # [batch_size, seq_length, hidden_dim]
       hsi_hidden = self.norm1(hsi_hidden)  # 新增：归一化

       lidar_hidden = F.relu(self.fc2(lidar_seq))  # [batch_size, seq_length, hidden_dim]
       lidar_hidden = self.norm2(lidar_hidden)  # 新增：归一化
       hsi_mamba_hidden = F.relu(self.fc3(hsi_mamba))  # [batch_size, seq_length, hidden_dim]
       hsi_mamba_hidden = self.norm3(hsi_mamba_hidden)  # 新增：归一化

       # 将三个隐藏特征在特征维度上拼接
       combined_hidden = torch.cat([hsi_hidden, lidar_hidden, hsi_mamba_hidden], dim=-1)  # [batch_size, seq_length, hidden_dim * 3]
       #print("combined_hidden", combined_hidden.shape)

       # 调整特征维度
       combined_hidden = F.relu(self.adjust_dim(combined_hidden))  # [batch_size, seq_length, hidden_dim]
       #combined_hidden = self.norm_adjust(combined_hidden)  # 新增：归一化

       # 计算注意力权重
       attention_weights = self.attention(combined_hidden)  # [batch_size, seq_length, 3]
       attention_weights = F.softmax(attention_weights, dim=-1)  # 归一化权重

       # 将注意力权重应用于每个模态的特征
       weighted_hsi = hsi_hidden * attention_weights[:, :, 0].unsqueeze(-1)
       weighted_lidar = lidar_hidden * attention_weights[:, :, 1].unsqueeze(-1)
       weighted_hsi_mamba = hsi_mamba_hidden * attention_weights[:, :, 2].unsqueeze(-1)

       # 融合后的特征
       fused_feature = weighted_hsi + weighted_lidar + weighted_hsi_mamba  # [batch_size, seq_length, hidden_dim]

       # 将融合后的特征映射回原始特征维度
       fused_feature = self.fc_out(fused_feature)  # [batch_size, seq_length, feature_dim]
       #fused_feature = self.norm_out(fused_feature)  # 新增：最终归一化

       return fused_feature


# class MultiModalAttentionFusion(nn.Module):
#     def __init__(self, feature_dim, hidden_dim):
#         super(MultiModalAttentionFusion, self).__init__()
#         self.feature_dim = feature_dim
#         self.hidden_dim = hidden_dim
        
#         # 调整特征维度的线性层
#         self.adjust_dim = nn.Linear(hidden_dim * 3, hidden_dim)
#         self.bn_adjust = nn.BatchNorm1d(hidden_dim)

#         # 线性变换层，将输入特征映射到隐藏空间
#         self.fc1 = nn.Linear(feature_dim, hidden_dim)
#         self.fc2 = nn.Linear(feature_dim, hidden_dim)
#         self.fc3 = nn.Linear(feature_dim, hidden_dim)

#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#         self.bn3 = nn.BatchNorm1d(hidden_dim)

#         # 注意力权重层
#         self.attention = nn.Linear(hidden_dim, 3)  # 输出3个权重，对应3个输入模态
#         self.bn_attention = nn.BatchNorm1d(3)

#         # 输出层
#         self.fc_out = nn.Linear(hidden_dim, feature_dim)
#         self.bn_out = nn.BatchNorm1d(feature_dim)

#     def forward(self, hsi_seq, lidar_seq, hsi_mamba):
#         # 输入特征的形状为 [batch_size, seq_length, feature_dim]
#         batch_size, seq_length, _ = hsi_seq.size()

#         # 将输入特征映射到隐藏空间
#         hsi_hidden = self.fc1(hsi_seq)  # [batch_size, seq_length, hidden_dim]
#         lidar_hidden = self.fc2(lidar_seq)  # [batch_size, seq_length, hidden_dim]
#         hsi_mamba_hidden = self.fc3(hsi_mamba)  # [batch_size, seq_length, hidden_dim]

#         # 调整形状以应用 BatchNorm1d
#         hsi_hidden = hsi_hidden.view(-1, self.hidden_dim)  # [batch_size * seq_length, hidden_dim]
#         lidar_hidden = lidar_hidden.view(-1, self.hidden_dim)  # [batch_size * seq_length, hidden_dim]
#         hsi_mamba_hidden = hsi_mamba_hidden.view(-1, self.hidden_dim)  # [batch_size * seq_length, hidden_dim]

#         # 应用 BatchNorm1d
#         hsi_hidden = self.bn1(hsi_hidden)
#         lidar_hidden = self.bn2(lidar_hidden)
#         hsi_mamba_hidden = self.bn3(hsi_mamba_hidden)

#         # 恢复形状
#         hsi_hidden = F.relu(hsi_hidden.view(batch_size, seq_length, self.hidden_dim))  # [batch_size, seq_length, hidden_dim]
#         lidar_hidden = F.relu(lidar_hidden.view(batch_size, seq_length, self.hidden_dim))  # [batch_size, seq_length, hidden_dim]
#         hsi_mamba_hidden = F.relu(hsi_mamba_hidden.view(batch_size, seq_length, self.hidden_dim))  # [batch_size, seq_length, hidden_dim]

#         # 将三个隐藏特征在特征维度上拼接
#         combined_hidden = torch.cat([hsi_hidden, lidar_hidden, hsi_mamba_hidden], dim=-1)  # [batch_size, seq_length, hidden_dim * 3]
#         #print("combined_hidden", combined_hidden.shape)

#         combined_hidden = F.relu(self.adjust_dim(combined_hidden))  # [batch_size, seq_length, hidden_dim]

#         # 调整形状以应用 BatchNorm1d
#         #combined_hidden = self.adjust_dim(combined_hidden).view(-1, self.hidden_dim)  # [batch_size * seq_length, hidden_dim]
#         #combined_hidden = self.bn_adjust(combined_hidden)  # 应用 BatchNorm1d
#         #combined_hidden = combined_hidden.view(batch_size, seq_length, self.hidden_dim)  # 恢复形状

#         #combined_hidden = F.relu(combined_hidden)

#         # 计算注意力权重
#         attention_weights = self.attention(combined_hidden)  # [batch_size, seq_length, 3]

#         # 调整形状以应用 BatchNorm1d
#         #attention_weights = attention_weights.view(-1, 3)  # [batch_size * seq_length, 3]
#         #attention_weights = self.bn_attention(attention_weights)  # 应用 BatchNorm1d
#         #attention_weights = attention_weights.view(batch_size, seq_length, 3)  # 恢复形状

#         attention_weights = F.softmax(attention_weights, dim=-1)  # 归一化权重

#         # 将注意力权重应用于每个模态的特征
#         weighted_hsi = hsi_hidden * attention_weights[:, :, 0].unsqueeze(-1)
#         weighted_lidar = lidar_hidden * attention_weights[:, :, 1].unsqueeze(-1)
#         weighted_hsi_mamba = hsi_mamba_hidden * attention_weights[:, :, 2].unsqueeze(-1)

#         # 融合后的特征
#         fused_feature = weighted_hsi + weighted_lidar + weighted_hsi_mamba  # [batch_size, seq_length, hidden_dim]

#         # 将融合后的特征映射回原始特征维度
#         fused_feature = self.fc_out(fused_feature)  # [batch_size, seq_length, feature_dim]

#         #fused_feature = fused_feature.view(-1, self.feature_dim)
#         #fused_feature = self.bn_out(fused_feature)
#         #fused_feature = fused_feature.view(batch_size, seq_length, self.feature_dim)

#         return fused_feature


class Fusion_stage1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #维度减半stride=2,padding=1；维度不变stride=1,padding=0
        self.conv1d_self = OneDConvNet(in_channels, out_channels, kernel_size=3, stride = 1, padding=1)

        # 添加BN层
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels * 2)  # x12的通道数是两倍

        self.fusion1_lamda1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.fusion1_lamda2 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda

        self.token_proj = nn.Conv1d(192, 192, kernel_size=2, stride=1, padding=0)######192,192,...
        
    def forward(self, x1, x2, x3):
        # 假设 x1 和 x2 的形状为 (batch_size, channels, height, width)
        # 首先，对 x1 和 x2 进行 1D 卷积 B, 64, 192
        # 假设输入为 (batch_size, 65, d_model)，实际有效长度为64
        x1 = self.token_proj(x1.transpose(1, 2)).transpose(1, 2)  # (B,65,C) -> (B,64,C)
        x2 = self.token_proj(x2.transpose(1, 2)).transpose(1, 2)
        x3 = self.token_proj(x3.transpose(1, 2)).transpose(1, 2)
        
        x1 = x1.permute(0, 2, 1) 
        x2 = x2.permute(0, 2, 1)  
        x3 = x3.permute(0, 2, 1)
        x1 = self.conv1d_self(x1)
        x1 = self.bn1(x1)
        x2 = self.conv1d_self(x2)
        x2 = self.bn2(x2)
        x3 = self.conv1d_self(x3)
        x3 = self.bn1(x3)
        x12 = self.fusion1_lamda1 * x1 + self.fusion1_lamda2 * x2
        x123 = torch.cat((x3, x12), dim=1)

        #x1 = self.conv1d_self(x1)
        #x1 = self.bn1(x1)
        #x2 = self.conv1d_self(x2)
        #x2 = self.bn2(x2)
        #x12 = torch.cat((x1, x2), dim=1)
        #x123 = self.fusion1_lamda1 * x12 + self.fusion1_lamda2 * x3
        x123 = self.bn3(x123)
        x123 = x123.permute(0, 2, 1)

        return x123

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        """
        初始化 Cross Attention 模块
        参数:
            d_model: 输入的特征维度
            n_heads: 多头注意力的头数
        """
        super(CrossAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 定义 Q、K、V 的线性变换层
        self.W_q = nn.Linear(d_model, d_model)  # Query 的线性变换
        self.W_k = nn.Linear(d_model, d_model)  # Key 的线性变换
        self.W_v = nn.Linear(d_model, d_model)  # Value 的线性变换
        self.W_o = nn.Linear(d_model, d_model)  # 输出线性变换

        self.bn_q = nn.BatchNorm1d(d_model)
        self.bn_k = nn.BatchNorm1d(d_model)
        self.bn_v = nn.BatchNorm1d(d_model)
        self.bn_o = nn.BatchNorm1d(d_model)

    def forward(self, query, key, value, mask=None):
        """
        前向传播
        参数:
            query: 查询序列，形状 [batch_size, query_len, d_model]
            key: 键序列，形状 [batch_size, key_len, d_model]
            value: 值序列，形状 [batch_size, key_len, d_model]
            mask: 可选的注意力掩码，形状 [batch_size, query_len, key_len]
        返回:
            输出: 经过 Cross Attention 的结果，形状 [batch_size, query_len, d_model]
        """
        batch_size = query.size(0)

        # 1. 线性变换生成 Q、K、V
        Q = self.W_q(query)  # [batch_size, query_len, d_model]
        #Q = self.bn_q(Q)  # 添加 BN
        #Q = F.relu(Q)     # 添加激活函数
        
        K = self.W_k(key)    # [batch_size, key_len, d_model]
        #K = self.bn_k(K)  # 添加 BN
        #K = F.relu(K)     # 添加激活函数
        
        V = self.W_v(value)  # [batch_size, key_len, d_model]
        #V = self.bn_v(V)  # 添加 BN
        #V = F.relu(V)     # 添加激活函数

        # 2. 将 Q、K、V 分成多头
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # Q, K, V 的形状变为 [batch_size, n_heads, seq_len, d_k]

        # 3. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # scores 形状: [batch_size, n_heads, query_len, key_len]

        # 4. 如果有掩码，应用掩码（比如在解码器中避免关注未来位置）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 5. 应用 Softmax 得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 6. 用注意力权重加权 Value
        attn_output = torch.matmul(attn_weights, V)
        # attn_output 形状: [batch_size, n_heads, query_len, d_k]

        # 7. 合并多头结果
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # 形状变为 [batch_size, query_len, d_model]
        #attn_output = self.bn_o(attn_output)  # 添加 BN
        #attn_output = F.relu(attn_output)     # 添加激活函数

        # 8. 最后通过线性层输出
        #output = self.W_o(attn_output)

        return attn_output#, attn_weights  # 返回输出和注意力权重（用于可视化或调试）

class Fusion_stage2(nn.Module):
    """
    通过加法融合两个序列的类
    """
    def __init__(self):
        super(Fusion_stage2, self).__init__()

        self.fusion2_lamda1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.fusion2_lamda2 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda

    def forward(self, seq1, seq2):
        """
        前向传播函数，将两个序列逐元素相加
        参数:
            seq1: 第一个序列，形状为 (B, L, D)
            seq2: 第二个序列，形状为 (B, L, D)
        返回:
            融合后的序列，形状为 (B, L, D)
        """
        # 确保两个序列的形状相同
        assert seq1.shape == seq2.shape, "两个序列的形状必须相同"
        
        # 逐元素相加
        fused_seq = self.fusion2_lamda1 * seq1 + self.fusion2_lamda2 * seq2
        return fused_seq


class Classifier(nn.Module):
    def __init__(self, input_dim=48, num_classes=15, sequence_length=65):
        super().__init__()
        # Input shape: (B, L, C)
        self.sequence_length = sequence_length
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim * sequence_length),
            nn.Linear(input_dim * sequence_length, 512),
            # nn.LayerNorm(512),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )
        
        # 1. 1D卷积保持序列维度
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_classes,
            kernel_size=3,
            padding=1
        )
        
        # 2. 动态位置编码（可选）
        self.position_embed = nn.Embedding(sequence_length, input_dim)

        self.bn_conv = nn.BatchNorm1d(num_classes)

        # self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        # 输入形状: (B, L, C)
        batch_size, seq_len, channels = x.size()
        #print("x:", x.shape)
        
        # x = x.mean(dim=1)  # 对序列维度取平均，形状变为 (B, C)
        # x = self.fc(x)  

        # x形状: (B, L, C)
        B = x.shape[0]
        #x = x.view(B, -1)  # 展平为(B, L*C)
        x = x.reshape(B, -1)
        return self.fc(x)

class SequenceToImage(nn.Module):
    def __init__(self, input_length=64, num_classes=15):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, x):
        """
        输入: [batch, 64] (展平后的8x8图像)
        输出: [batch, 8, 8] (还原后的分类结果图像)
        """
        batch_size = x.shape[0]
        
        # 转换为图像形状
        pred_image = x.view(batch_size, num_clas)  # [batch, 8, 8]
        
        return pred_image


