import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from resnet import ResNet18
from scipy.fft import fft
from MACAM import MCAM

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransZero(nn.Module):
    def __init__(self, att, init_w2v_att, seenclass, unseenclass,
                 is_bias=True, bias=1, is_conservative=True):
        super(TransZero, self).__init__()
        self.dim_f = 512
        self.dim_v = 768  # 768
        self.nclass = 16
        self.seenclass = seenclass
        self.unseenclass = unseenclass
        self.is_bias = is_bias
        self.is_conservative = is_conservative
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        # class-level semantic vectors
        self.att = nn.Parameter(F.normalize(att), requires_grad=False)  # 类别向量（对角线为一的矩阵）
        # GloVe features for attributes name
        self.V = nn.Parameter(F.normalize(init_w2v_att), requires_grad=True)  # 语义向量
        # for self-calibration
        self.bias = nn.Parameter(torch.tensor(bias), requires_grad=False)
        mask_bias = np.ones((1, self.nclass))
        mask_bias[:, self.seenclass.cpu().numpy()] *= -1
        self.mask_bias = nn.Parameter(torch.tensor(
            mask_bias, dtype=torch.float), requires_grad=False)
        # mapping
        self.W_1 = nn.Parameter(nn.init.normal_(
            torch.empty(self.dim_v, 512)), requires_grad=True)

        # transformer
        self.transformer = Transformer(
            ec_layer=1,
            dc_layer=1,
            dim_com=512,
            dim_feedforward=512,
            dropout=0.4,
            SAtt=True,
            heads=1,
            aux_embed=True)

        # 初始化ResNet18特征提取器
        self.resnet18 = ResNet18(num_c=16)
        self.resnet18_spectrogram = ResNet18(num_c=16)  # 用于频谱图
        self.MCAM = MCAM(in_channels=512)

        # for loss computation
        self.log_softmax_func = nn.LogSoftmax(dim=1)
        self.weight_ce = nn.Parameter(torch.eye(self.nclass), requires_grad=False)

    def forward(self, input, from_imu=True):
        """
        前向传播函数。

        参数:
        input (torch.Tensor): 输入数据
        from_imu (bool): 是否从IMU数据提取特征

        返回:
        dict: 包含预测结果和嵌入特征的字典
        """
        if from_imu:
            input = input.cpu().numpy()  # (22, 170, 27)
            input_imu = np.transpose(input, (0, 2, 1))  # (22, 27, 170)
            input_imu = torch.from_numpy(input_imu).float().to(self.device)
            Fs_imu,_ = self.resnet18(input_imu)
            # Fs = Fs_imu

            # 生成频谱图并处理
            input_spectrogram = self.compute_spectrogram(input)
            input_spectrogram = torch.from_numpy(input_spectrogram).float().to(self.device)
            Fs_spectrogram, _ = self.resnet18_spectrogram(input_spectrogram)

            # 特征融合
            Fs = self.MCAM(Fs_imu, Fs_spectrogram)
        else:
            Fs = input

        # transformer-based visual-to-semantic embedding
        Fs = Fs.unsqueeze(3)  # torch.Size([22, 512, 1, 1])
        v2s_embed = self.forward_feature_transformer(Fs)
        # classification
        package = {'pred': self.forward_attribute(v2s_embed),  # 计算嵌入特征的类别预测概率
                   'embed': v2s_embed}
        package['S_pp'] = package['pred']
        return package

    def compute_spectrogram(self, input):
        """
        计算IMU数据的频谱图，并进行归一化处理。

        参数:
        input (numpy.ndarray): 输入IMU数据，形状为 (22, 170, 27)

        返回:
        numpy.ndarray: 归一化后的频谱图，形状为 (22, 27, 85)
        """
        input_spectrogram = np.zeros((input.shape[0], input.shape[2], input.shape[1] // 2 + 1))
        for i in range(input.shape[0]):
            for j in range(input.shape[2]):
                spectrum = np.abs(fft(input[i, :, j]))[:input.shape[1] // 2 + 1]
                # 归一化操作
                spectrum_min = np.min(spectrum)
                spectrum_max = np.max(spectrum)
                if spectrum_max > spectrum_min:
                    spectrum = (spectrum - spectrum_min) / (spectrum_max - spectrum_min)
                else:
                    spectrum = np.zeros_like(spectrum)  # 防止除零错误
                input_spectrogram[i, j, :] = spectrum
        return input_spectrogram

    def forward_feature_transformer(self, Fs):
        """
        基于transformer的特征转换。

        参数:
        Fs (torch.Tensor): 输入特征

        返回:
        torch.Tensor: 转换后的嵌入特征
        """
        # visual
        if len(Fs.shape) == 4:
            shape = Fs.shape
            Fs = Fs.reshape(shape[0], shape[1], shape[2] * shape[3])
        Fs = F.normalize(Fs, dim=1)  # 标准化处理
        V_n = self.V  # attributes
        # V_n = F.normalize(self.V)
        # locality-augmented visual features
        Trans_out = self.transformer(Fs, V_n)  # 实现了信息和语义信息的融合，从而提升模型的特征表示能力
        # embedding to semantic space
        embed = torch.einsum('iv,vf,bif->bi', V_n, self.W_1, Trans_out)  # 对输入特征进行一种线性变换和聚合操作，以得到新的嵌入特征表示
        # 使用爱因斯坦求和约定 torch.einsum 计算特征嵌入
        return embed

    def forward_attribute(self, embed):  # 将嵌入特征与类别属性向量进行点积计算，并添加偏置修正
        """
        将嵌入特征与类别属性向量进行点积计算，并添加偏置修正。

        参数:
        embed (torch.Tensor): 嵌入特征

        返回:
        torch.Tensor: 类别预测
        """
        embed = torch.einsum('ki,bi->bk', self.att, embed)
        self.vec_bias = self.mask_bias * self.bias
        embed = embed + self.vec_bias
        return embed

    def compute_loss_Self_Calibrate(self, in_package):  # 自校准损失确保模型对未见类的预测具有合理的置信度
        S_pp = in_package['pred']
        Prob_all = F.softmax(S_pp, dim=-1)
        Prob_unseen = Prob_all[:, self.unseenclass]
        assert Prob_unseen.size(1) == len(self.unseenclass)
        mass_unseen = torch.sum(Prob_unseen, dim=1)
        loss_pmp = -torch.log(torch.mean(mass_unseen))
        return loss_pmp

    def compute_aug_cross_entropy(self, in_package):  # 增强的交叉熵损失用于优化分类准确性
        Labels = in_package['batch_label']
        S_pp = in_package['pred']

        if self.is_bias:
            S_pp = S_pp - self.vec_bias

        if not self.is_conservative:
            S_pp = S_pp[:, self.seenclass]
            Labels = Labels[:, self.seenclass]
            assert S_pp.size(1) == len(self.seenclass)

        Prob = self.log_softmax_func(S_pp)

        loss = -torch.einsum('bk,bk->b', Prob, Labels)
        loss = torch.mean(loss)
        return loss

    def compute_reg_loss(self, in_package):  # 正则化损失保证嵌入特征与目标属性向量的匹配，防止过拟合
        tgt = torch.matmul(in_package['batch_label'], self.att)
        embed = in_package['embed']
        loss_reg = F.mse_loss(embed, tgt, reduction='mean')
        return loss_reg

    def compute_loss(self, in_package):
        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]

        loss_CE = self.compute_aug_cross_entropy(in_package)
        loss_cal = self.compute_loss_Self_Calibrate(in_package)
        loss_reg = self.compute_reg_loss(in_package)

        loss = loss_CE  + 0.005 * loss_reg + 0.3 * loss_cal

        out_package = {'loss': loss, 'loss_CE': loss_CE,
                       'loss_cal': loss_cal, 'loss_reg': loss_reg}
        return out_package


class Transformer(nn.Module):
    # dim_feedforward = 512
    # in_dim_cv = 512
    def __init__(self, ec_layer=1, dc_layer=1, dim_com=768,
                 dim_feedforward=512, dropout=0.1, heads=1,
                 in_dim_cv=512, in_dim_attr=768, SAtt=True,
                 aux_embed=True):
        super(Transformer, self).__init__()
        # input embedding
        self.embed_cv = nn.Sequential(nn.Linear(in_dim_cv, dim_com))
        if aux_embed:
            self.embed_cv_aux = nn.Sequential(nn.Linear(in_dim_cv, dim_com))
        self.embed_attr = nn.Sequential(nn.Linear(in_dim_attr, dim_com))
        # transformer encoder
        self.transformer_encoder = MultiLevelEncoder_woPad(N=ec_layer,
                                                           d_model=dim_com,
                                                           h=1,
                                                           d_k=dim_com,
                                                           d_v=dim_com,
                                                           d_ff=dim_feedforward,
                                                           dropout=dropout)
        # transformer decoder
        decoder_layer = TransformerDecoderLayer(d_model=dim_com,
                                                nhead=heads,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                SAtt=SAtt)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=dc_layer)

    def forward(self, f_cv, f_attr):
        # linearly map to common dim
        h_cv = self.embed_cv(f_cv.permute(0, 2, 1))  # feature和att通过线性层映射到公共的维度空间
        h_attr = self.embed_attr(f_attr)
        h_attr_batch = h_attr.unsqueeze(0).repeat(f_cv.shape[0], 1, 1)
        # visual encoder
        memory = self.transformer_encoder(h_cv).permute(1, 0, 2)  # 通过 Transformer 编码器对feature进行编码
        # attribute-visual decoder
        out = self.transformer_decoder(h_attr_batch.permute(1, 0, 2), memory)  # att和feature通过 Transformer 解码器进行融合
        # 解码器中，进行多头注意力机制的交互，最终输出融合后的特征表示 out
        return out.permute(1, 0, 2)  # 将解码器输出的结果out返回，并调整维度


class EncoderLayer(nn.Module):
    # d_ff = 2048
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=512,
                 dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadGeometryAttention(d_model, d_k, d_v, h, dropout,
                                                identity_map_reordering=identity_map_reordering,
                                                attention_module=attention_module,
                                                attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(
            d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights,
                attention_mask=None, attention_weights=None, pos=None):
        q, k = (queries + pos, keys +
                pos) if pos is not None else (queries, keys)
        att = self.mhatt(q, k, values, relative_geometry_weights,
                         attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder_woPad(nn.Module):
    # d_ff = 2048
    def __init__(self, N, d_model=512, d_k=64, d_v=64, h=8, d_ff=512,
                 dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder_woPad, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

        self.WGs = nn.ModuleList(
            [nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, input, attention_mask=None, attention_weights=None, pos=None):

        relative_geometry_embeddings = BoxRelationalEmbedding(
            # grid_size=(14, 14)
            input, grid_size=(1, 1))
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(
            -1, 64)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [layer(
            flatten_relative_geometry_embeddings).view(box_size_per_head) for layer in self.WGs]
        relative_geometry_weights = torch.cat(
            (relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)
        out = input
        for layer in self.layers:
            out = layer(out, out, out, relative_geometry_weights,
                        attention_mask, attention_weights, pos=pos)
        return out


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    # dim_feedforward = 2048,
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1,
                 activation="relu", SAtt=True):
        super(TransformerDecoderLayer, self).__init__(d_model, nhead,
                                                      dim_feedforward=dim_feedforward,
                                                      dropout=dropout,
                                                      activation=activation)
        self.SAtt = SAtt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if self.SAtt:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def get_relative_pos(x, batch_size, norm_len):
    x = x.view(1, -1, 1).expand(batch_size, -1, -1)
    return x / norm_len


def get_grids_pos(batch_size, seq_len, grid_size=(7, 7)):
    # print(seq_len)
    # print(grid_size[0] * grid_size[1])
    assert seq_len == grid_size[0] * grid_size[1]
    x = torch.arange(0, grid_size[0]).float().cuda()
    y = torch.arange(0, grid_size[1]).float().cuda()
    px_min = x.view(-1, 1).expand(-1, grid_size[0]).contiguous().view(-1)
    py_min = y.view(1, -1).expand(grid_size[1], -1).contiguous().view(-1)
    px_max = px_min + 1
    py_max = py_min + 1
    rpx_min = get_relative_pos(px_min, batch_size, grid_size[0])
    rpy_min = get_relative_pos(py_min, batch_size, grid_size[1])
    rpx_max = get_relative_pos(px_max, batch_size, grid_size[0])
    rpy_max = get_relative_pos(py_max, batch_size, grid_size[1])
    return rpx_min, rpy_min, rpx_max, rpy_max


def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True,
                           grid_size=(7, 7)):
    batch_size, seq_len = f_g.size(0), f_g.size(1)
    x_min, y_min, x_max, y_max = get_grids_pos(batch_size, seq_len, grid_size)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)
    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)
    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))
    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)
    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)
    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))
        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(
            batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat
        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)


class ScaledDotProductGeometryAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        super(ScaledDotProductGeometryAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()
        self.comment = comment

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, box_relation_embed_matrix,
                attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h,
                                    self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h,
                                   self.d_v).permute(0, 2, 1, 3)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        w_g = box_relation_embed_matrix
        w_a = att
        w_mn = - w_g + w_a
        w_mn = torch.softmax(w_mn, -1)
        att = self.dropout(w_mn)
        out = torch.matmul(att, v).permute(
            0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class MultiHeadGeometryAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False,
                 can_be_stateful=False, attention_module=None,
                 attention_module_kwargs=None, comment=None):
        super(MultiHeadGeometryAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductGeometryAttention(
            d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, relative_geometry_weights,
                attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys
            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values
        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, relative_geometry_weights,
                                 attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, relative_geometry_weights,
                                 attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out


class PositionWiseFeedForward(nn.Module):
    # d_ff = 2048
    def __init__(self, d_model=512, d_ff=512, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out


if __name__ == '__main__':
    pass
