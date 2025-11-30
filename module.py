# -*- coding:utf-8 -*-
# @Time       :2025/6
# @AUTHOR     :liubaiyan
# @FileName   :module.py
from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F

from structure import cnn_encoder_A_and_B, cnnModule, SimAM, ConvToSequence, MambaVisionMixer, Fusion_stage1, CrossAttention, Fusion_stage2, Classifier, SequenceToImage
from structure import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MineNet(nn.Module):
    """3-layer MLP discriminator for MINE."""
    def __init__(self, d=128):
        super().__init__()
        self.T = nn.Sequential(
            nn.Linear(d*2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, u, v):          # u,v: [B*L, D]
        uv = torch.cat([u, v], dim=-1).to(device)
        # print("uv:", uv.shape)
        self.T = self.T.to(device)
        return self.T(uv).squeeze()   # [B*L]

d_feat = 64                     # 你的特征维
mine_net = MineNet(d=d_feat) 
def mutual_information(cnn_to_seq, spectral_to_seq):
    """
    cnn_to_seq, spectral_to_seq: [B, L, D]
    returns scalar MI estimate (lower bound)
    """
    global mine_net
    B, L, D = cnn_to_seq.shape
    u = cnn_to_seq.reshape(-1, D).to(device)        # [B*L, D]
    v = spectral_to_seq.reshape(-1, D).to(device) 
    # print("u", u.shape, "v", v.shape)   # 2080, 64

    # positive pairs
    t_pos = mine_net(u, v)               # [B*L]

    # negative pairs (shuffle v)
    v_shuf = v[torch.randperm(v.size(0))]
    t_neg = mine_net(u, v_shuf)

    # Eq.(6)  用移动平均估计分母
    with torch.no_grad():
        mi_lb = t_pos.mean() - (t_neg.exp().mean() + 1e-8).log()
    return mi_lb

import torch


def redundancy_loss(a, b):
    #b = b[:, 1:65, :]  # 直接截断填充部分
    #print("b:", b.shape)
    #print("a:", a.shape)

    a, b = a.reshape(a.size(0), -1), b.reshape(b.size(0), -1)
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    corr = (a * b).mean(1) / (a.std(1) * b.std(1) + 1e-8)
    return corr.abs().mean()

class module_fusion(nn.Module):
    def __init__(self, l1, l2, maba_input_dim, fusion1_dim, across_heads):
        super().__init__()

        # 四个标量系数
        self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.beta   = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.gamma  = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        #self.alpha_mi = nn.Parameter(torch.tensor(0.05))

        self.cnnHSI = cnnModule(64, 32) #cnn输出通道数为64
        self.cnnLiDAR = cnnModule(64, 32) #cnn输出通道数为32
        # self.cnnSpectral = cnnModule1D(64, 32) #cnn输出通道数为32

        self.MultiCNN = cnn_encoder_A_and_B(l1, l2)
        # self.sepctralCNN = SpectralExtractor(l1,out_channels=64)#输出是32+16+16=64#改回来直接就是64
        self.sepctralCNN = EnhancedSpectralExtractor(l1,out_channels=64)

        self.add2 = weight_add(in_channels=96, out_channels=64)

        self.attention3d = SimAM()

        #self.HSItoSequence = ConvToSequenceSin(192, 192)#(8, 192)#(8, maba_input_dim)
        #self.LiDARtoSequence = ConvToSequenceSin(192, 192)#(8, 192)#(8, 192)
        self.toSequence = ConvToSequence(8, 64)#(8, 192)#(8, maba_input_dim)

        self.mambaCNN = MambaVisionMixer(
            d_model=maba_input_dim,  # 输入维度
        )
        self.mambaSep = MambaVisionMixer(
            d_model=maba_input_dim,  # 输入维度
        )

        #self.fusion1 = Fusion_stage1(192, 96)
        self.fusion1 = MultiModalAttentionFusion(64, hidden_dim=fusion1_dim)

        self.crossAtten = CrossAttention(d_model=maba_input_dim, n_heads=across_heads)

        self.fusion2 = Fusion_stage2()

        self.classifier = Classifier(input_dim=maba_input_dim, num_classes=15, sequence_length=65)
        self.output_classifier = Classifier(input_dim=maba_input_dim, num_classes=15, sequence_length=65)

    def forward(self, img11, img21, img12, img22, img13, img23):
        # 通过 cnnHSI 模块
        #cnn_output_HSI = self.cnnHSI(img11)####################
        # 通过 cnnLiDAR 模块
        #cnn_output_LiDAR = self.cnnLiDAR(img21)#######################

        cnn_output_HSI, cnn_output_LiDAR, = self.MultiCNN(img11, img21, img12, img22, img13, img23)
        # print("cnn_output_HSI:", cnn_output_HSI.shape, "cnn_output_LiDAR", cnn_output_LiDAR.shape)
        cnn_Sepctral = self.sepctralCNN(img11)##########################
        # print("cnn_Sepctral:", cnn_Sepctral.shape)
        
        cnn_output_HSI = self.cnnHSI(cnn_output_HSI)
        cnn_output_LiDAR = self.cnnLiDAR(cnn_output_LiDAR)
        # cnn_Sepctral = self.cnnSpectral(cnn_Sepctral)
        # print("cnn_output_HSI:", cnn_output_HSI.shape, "cnn_output_LiDAR", cnn_output_LiDAR.shape)
        mi_hsi_lidar = 0#mutual_information_3d(cnn_output_HSI, cnn_output_LiDAR)
        cnn_fusion = self.add2(cnn_output_HSI, cnn_output_LiDAR)
        # mi_fusion_hsi = mutual_information_3d(cnn_fusion, self.mi_conv(cnn_output_HSI))
        # mi_fusion_lidar = mutual_information_3d(cnn_fusion, self.mi_conv(cnn_output_LiDAR))
        # print("cnn_fusion:", cnn_fusion.shape)

        # 通过 SimAM 模块
        cnn_fusion_atten = self.attention3d(cnn_fusion, lambda_SimAM = 0.05)
        cnn_Sepctral = self.attention3d(cnn_Sepctral, lambda_SimAM = 0.05)
        # print("cnn_fusion_atten:", cnn_fusion_atten.shape)

        cnn_to_seq = self.toSequence(cnn_fusion_atten)
        sepctral_to_seq = self.toSequence(cnn_Sepctral)
        # print("cnn_to_seq:", cnn_to_seq.shape, "sepctral_to_seq", sepctral_to_seq.shape)

        mi_seq = mutual_information(cnn_to_seq, sepctral_to_seq)
        # print("mi_seq:", mi_seq)

        cnn_mamba = self.mambaCNN(cnn_to_seq)
        sepctral_mamba = self.mambaSep(sepctral_to_seq)
        # print("cnn_mamba:", cnn_mamba.shape, "sepctral_mamba", sepctral_mamba.shape)
        
        fusion_1_cnn = self.fusion1(cnn_to_seq, sepctral_to_seq, cnn_mamba)
        fusion_1_sepctral = self.fusion1(sepctral_to_seq, cnn_to_seq, sepctral_mamba)
        # print("fusion_1_cnn:", fusion_1_cnn.shape, "fusion_1_sepctral", fusion_1_sepctral.shape)

        attention_cnn = self.crossAtten(fusion_1_cnn, fusion_1_sepctral, fusion_1_sepctral)
        attention_sep = self.crossAtten(fusion_1_sepctral, fusion_1_cnn, fusion_1_cnn)
        # print("attention_cnn:", attention_cnn.shape, "attention_sep", attention_sep.shape)

        classify_cnn = self.classifier(fusion_1_cnn)
        classify_sep = self.classifier(fusion_1_sepctral)
        #print("Attention_output_HSI:", attention_HSI.shape, "Attention_output_LiDAR", attention_LiDAR.shape)
        
        fusuin_2 = self.fusion2(attention_cnn, attention_sep)
        #print("Fusion2_output:", fusuin_2.shape)

        output_classify = self.output_classifier(fusuin_2)
        #print("output_classify:", output_classify.shape)
        # output_classify = self.alpha * output_classify + self.beta * classify_cnn + self.gamma * classify_sep

        return output_classify, classify_cnn, classify_sep, mi_seq, mi_hsi_lidar
        
        
####################################################################

class MMA_test(nn.Module):
    def __init__(self, l1, l2, patch_size, num_patches, num_classes, encoder_embed_dim, decoder_embed_dim, en_depth, en_heads,
                 de_depth, de_heads, mlp_dim, dim_head=16, dropout=0., emb_dropout=0.,fusion='trans'):
        super().__init__()

        self.encoder = encoder_A_and_B(l1, l2,patch_size, num_patches,num_classes, encoder_embed_dim,  en_depth, en_heads,dim_head, mlp_dim, dropout, emb_dropout,fusion)
        self.classifier = classification(encoder_embed_dim,num_classes)


    def forward(self, img11, img21, img12, img22, img13, img23):

        x_vit, x_cnn,x1_out,x2_out,x1c_out,x_fuse1,x_fuse2,x_transfusion = self.encoder(img11, img21, img12, img22, img13, img23, single_FLAG=0)
        x_cls = self.classifier(x_vit, x_cnn)
        

        return x_cls