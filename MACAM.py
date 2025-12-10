import torch
import torch.nn as nn
import torch.nn.functional as F

class MCAM(nn.Module):
    def __init__(self, in_channels):
        super(MCAM, self).__init__()

        self.in_channels = in_channels

        # 1x1卷积层，用于从输入特征图中提取VOPT、QOPT和KOPT
        self.VOPT_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.QOPT_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.KOPT_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        # 1x1卷积层，用于从输入特征图中提取VSAR、QSAR和KSAR
        self.VSAR_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.QSAR_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.KSAR_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, imu_features, spectrum_features):
        batch_size = imu_features.size(0)

        # 提取频谱图像特征的VOPT、QOPT和KOPT
        VOPT = self.VOPT_conv(spectrum_features)
        QOPT = self.QOPT_conv(spectrum_features)
        KOPT = self.KOPT_conv(spectrum_features)

        # 提取IMU特征的VSAR、QSAR和KSAR
        VSAR = self.VSAR_conv(imu_features)
        QSAR = self.QSAR_conv(imu_features)
        KSAR = self.KSAR_conv(imu_features)

        # 计算频谱图像特征的自注意力图Shigh_OPT
        S_OPT = torch.softmax(torch.matmul(QOPT.permute(0, 2, 1), KOPT), dim=-1)

        # 计算IMU特征的自注意力图Shigh_SAR
        S_SAR = torch.softmax(torch.matmul(QSAR.permute(0, 2, 1), KSAR), dim=-1)

        # 计算频谱和IMU特征的交叉融合注意力图Shigh_cro
        S_cro = S_OPT * S_SAR

        # 使用频谱特征图VOPT加权得到注意力加权后的频谱特征图Atthigh_OPT
        Att_OPT = S_cro * VOPT

        # 使用IMU特征图VSAR加权得到注意力加权后的IMU特征图Atthigh_SAR
        Att_SAR = S_cro * VSAR

        # 将注意力加权后的频谱和IMU特征图进行叠加得到最终的联合注意力图Atthigh_OPT_SAR
        Att_OPT_SAR = Att_OPT * Att_SAR

        return Att_OPT_SAR


if __name__ == '__main__':
    model = MCAM(in_channels=512)
    model.train()
    imu_features = torch.randn(22, 512, 1)
    spectrum_features = torch.randn(22, 512, 1)
    print(model)
    print("input:", imu_features.shape, spectrum_features.shape)
    print("output:", model(imu_features, spectrum_features).shape)