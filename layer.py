import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv
import numpy as np
from graph_part.kan import KAN
import copy


## 多模态信息整合
class VariLengthInputLayer(nn.Module):
    def __init__(self, input_data_dims, d_k, d_v, n_head, dropout):
        super(VariLengthInputLayer, self).__init__()
        self.n_head = n_head
        self.dims = input_data_dims
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = []
        self.w_ks = []
        self.w_vs = []
        for i, dim in enumerate(self.dims):
            self.w_q = nn.Linear(dim, n_head * d_k, bias=False)
            self.w_k = nn.Linear(dim, n_head * d_k, bias=False)
            self.w_v = nn.Linear(dim, n_head * d_v, bias=False)
            self.w_qs.append(self.w_q)
            self.w_ks.append(self.w_k)
            self.w_vs.append(self.w_v)
            self.add_module('linear_q_%d_%d' % (dim, i), self.w_q)
            self.add_module('linear_k_%d_%d' % (dim, i), self.w_k)
            self.add_module('linear_v_%d_%d' % (dim, i), self.w_v)

        self.attention = Attention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.fc = nn.Linear(n_head * d_v, n_head * d_v)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_head * d_v, eps=1e-6)

    def forward(self, input_data, mask=None):
        """
        输入的向量是各个模态concatenate起来的
        """
        temp_dim = 0
        bs = input_data.size(0)
        modal_num = len(self.dims)
        q = torch.zeros(bs, modal_num, self.n_head * self.d_k).cuda()
        k = torch.zeros(bs, modal_num, self.n_head * self.d_k).cuda()
        v = torch.zeros(bs, modal_num, self.n_head * self.d_v).cuda()
        for i in range(modal_num):
            w_q = self.w_qs[i]
            w_k = self.w_ks[i]
            w_v = self.w_vs[i]

            data = input_data[:, temp_dim: temp_dim + self.dims[i]]
            temp_dim += self.dims[i]
            q[:, i, :] = w_q(data)
            k[:, i, :] = w_k(data)
            v[:, i, :] = w_v(data)

        q = q.view(bs, modal_num, self.n_head, self.d_k)
        k = k.view(bs, modal_num, self.n_head, self.d_k)
        v = v.view(bs, modal_num, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn, residual = self.attention(q, k, v)  # 注意因为没有同输入相比维度发生变化，因此以v作为残差
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        residual = residual.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn


## 注意力
class Attention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # softmax+dropout
        attn = attn / abs(attn.min())
        attn = self.dropout(F.softmax(F.normalize(attn, dim=-1), dim=-1))
        # attn = self.dropout(F.softmax(attn, dim=-1))
        # 概率分布xV
        output = torch.matmul(attn, v)

        return output, attn, v


class EncodeLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, dropout):
        super(EncodeLayer, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = Attention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, modal_num, mask=None):
        bs = q.size(0)
        residual = q
        q = self.w_q(q).view(bs, modal_num, self.n_head, self.d_k)
        k = self.w_k(k).view(bs, modal_num, self.n_head, self.d_k)
        v = self.w_v(v).view(bs, modal_num, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn, _ = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn


class OutputLayer(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes, modal_num, dropout=0.5):
        super(OutputLayer, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(d_hidden + modal_num ** 2, n_classes)

    def forward(self, x, attn_embedding):
        x = self.mlp_head(x)
        combined_x = torch.cat((x, attn_embedding), dim=-1)
        output = self.classifier(combined_x)
        return F.log_softmax(output, dim=1), combined_x


class OutputLayer_Filter(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes, modal_num, seq_len, dropout=0.5):
        super(OutputLayer_Filter, self).__init__()
        self.seq_len = seq_len
        self.filter = FilterLayer(hidden_size=d_in // seq_len, seq_len=seq_len, prob=dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(d_hidden + modal_num ** 2, n_classes)

    def forward(self, x, attn_embedding):
        batch_size, hidden_size = x.shape
        x = x.reshape(batch_size, self.seq_len, hidden_size // self.seq_len)
        x = self.filter(x)
        x = x.reshape(batch_size, hidden_size)
        x = self.mlp_head(x)
        combined_x = torch.cat((x, attn_embedding), dim=-1)
        output = self.classifier(combined_x)
        return F.log_softmax(output, dim=1), combined_x


class OutputFilterLayer1(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes, modal_num, dropout=0.5):
        super(OutputLayer, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(d_hidden, n_classes)

    def forward(self, x):
        x = self.mlp_head(x)
        output = self.classifier()
        return F.log_softmax(output, dim=1), x


class OutputLayer_KAN(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes, modal_num, dropout=0.5):
        super(OutputLayer_KAN, self).__init__()
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(d_in),
        #     nn.Linear(d_in, d_hidden),
        #     nn.GELU(),
        #     nn.Dropout(dropout)
        # )
        self.kan = KAN([d_in + modal_num ** 2, d_hidden, n_classes], base_activation=nn.GELU)
        # self.classifier = nn.Linear(d_hidden + modal_num ** 2, n_classes)

    def forward(self, x, attn_embedding):
        combined_x = torch.cat((x, attn_embedding), dim=-1)
        output = self.kan(combined_x)
        return F.log_softmax(output, dim=1), combined_x


class OutputLayer_KAN_1layer(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes, modal_num, dropout=0.5):
        super(OutputLayer_KAN_1layer, self).__init__()
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(d_in),
        #     nn.Linear(d_in, d_hidden),
        #     nn.GELU(),
        #     nn.Dropout(dropout)
        # )
        self.kan = KAN([d_in, d_hidden], base_activation=nn.GELU)
        self.classifier = nn.Linear(d_hidden + modal_num ** 2, n_classes)

    def forward(self, x, attn_embedding):
        x = self.kan(x)
        combined_x = torch.cat((x, attn_embedding), dim=-1)
        output = self.classifier(combined_x)
        return F.log_softmax(output, dim=1), combined_x


## 不使用模态特定信息
class OutputLayer2(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes, modal_num, dropout=0.5):
        super(OutputLayer2, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(d_hidden, n_classes)

    def forward(self, x, attn_embedding):
        x = self.mlp_head(x)
        # combined_x = torch.cat((x, attn_embedding), dim=-1)
        output = self.classifier(x)
        return F.log_softmax(output, dim=1), x


class FeedForwardLayer(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        # 两个fc层，对最后的512维度进行变换
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


# 多模态注意力
class VLTransformer(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(VLTransformer, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm['n_hidden']
        self.d_k = hyperpm['n_hidden']
        self.d_v = hyperpm['n_hidden']
        self.n_head = hyperpm['n_head']
        self.dropout = hyperpm['fusion_dropout']
        self.n_layer = hyperpm['n_layer']
        self.modal_num = hyperpm['modal_num']
        self.n_class = hyperpm['num_classes']
        self.d_out = self.d_v * self.n_head * self.modal_num  # 24*4*2

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []

        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            # encoder = nn.MultiheadAttention(self.d_k * self.n_head, self.n_head, dropout = self.dropout) #nn.multi_head_attn
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer(d_in, self.d_v * self.n_head, self.n_class, self.modal_num, self.dropout)

    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())

        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            # x = x.transpose(1, 0)#nn.multi_head_attn
            # x, attn = self.Encoder[i](x, x, x)#nn.multi_head_attn
            # x = x.transpose(1, 0)#nn.multi_head_attn
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())

        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1)
        output, hidden = self.Outputlayer(x, attn_embedding)
        return output, hidden, attn_map


# 多模态注意力——去掉模态特定信息
class VLTransformer2(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(VLTransformer2, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm['n_hidden']
        self.d_k = hyperpm['n_hidden']
        self.d_v = hyperpm['n_hidden']
        self.n_head = hyperpm['n_head']
        self.dropout = hyperpm['fusion_dropout']
        self.n_layer = hyperpm['n_layer']
        self.modal_num = hyperpm['modal_num']
        self.n_class = hyperpm['num_classes']
        self.d_out = self.d_v * self.n_head * self.modal_num  # 24*4*2

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []

        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            # encoder = nn.MultiheadAttention(self.d_k * self.n_head, self.n_head, dropout = self.dropout) #nn.multi_head_attn
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer2(d_in, self.d_v * self.n_head, self.n_class, self.modal_num, self.dropout)

    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())

        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            # x = x.transpose(1, 0)#nn.multi_head_attn
            # x, attn = self.Encoder[i](x, x, x)#nn.multi_head_attn
            # x = x.transpose(1, 0)#nn.multi_head_attn
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())

        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1)
        output, hidden = self.Outputlayer(x, attn_embedding)
        return output, hidden, attn_map


class VLTransformer_KAN(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(VLTransformer_KAN, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm['n_hidden']
        self.d_k = hyperpm['n_hidden']
        self.d_v = hyperpm['n_hidden']
        self.n_head = hyperpm['n_head']
        self.dropout = hyperpm['fusion_dropout']
        self.n_layer = hyperpm['n_layer']
        self.modal_num = hyperpm['modal_num']
        self.n_class = hyperpm['num_classes']
        self.d_out = self.d_v * self.n_head * self.modal_num  # 24*4*2

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []

        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            # encoder = nn.MultiheadAttention(self.d_k * self.n_head, self.n_head, dropout = self.dropout) #nn.multi_head_attn
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer_KAN(d_in, self.d_v * self.n_head, self.n_class, self.modal_num, self.dropout)

    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())

        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            # x = x.transpose(1, 0)#nn.multi_head_attn
            # x, attn = self.Encoder[i](x, x, x)#nn.multi_head_attn
            # x = x.transpose(1, 0)#nn.multi_head_attn
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())

        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1)
        output, hidden = self.Outputlayer(x, attn_embedding)
        return output, hidden, attn_map


class VLTransformer_KAN_1layer(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(VLTransformer_KAN_1layer, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm['n_hidden']
        self.d_k = hyperpm['n_hidden']
        self.d_v = hyperpm['n_hidden']
        self.n_head = hyperpm['n_head']
        self.dropout = hyperpm['fusion_dropout']
        self.n_layer = hyperpm['n_layer']
        self.modal_num = hyperpm['modal_num']
        self.n_class = hyperpm['num_classes']
        self.d_out = self.d_v * self.n_head * self.modal_num  # 24*4*2

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []

        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            # encoder = nn.MultiheadAttention(self.d_k * self.n_head, self.n_head, dropout = self.dropout) #nn.multi_head_attn
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer_KAN_1layer(d_in, self.d_v * self.n_head, self.n_class, self.modal_num,
                                                  self.dropout)

    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())

        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            # x = x.transpose(1, 0)#nn.multi_head_attn
            # x, attn = self.Encoder[i](x, x, x)#nn.multi_head_attn
            # x = x.transpose(1, 0)#nn.multi_head_attn
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())

        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1)
        output, hidden = self.Outputlayer(x, attn_embedding)
        return output, hidden, attn_map


class VLTransformer_FilterLayer(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(VLTransformer_FilterLayer, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm['n_hidden']
        self.d_k = hyperpm['n_hidden']
        self.d_v = hyperpm['n_hidden']
        self.n_head = hyperpm['n_head']
        self.dropout = hyperpm['fusion_dropout']
        self.n_layer = hyperpm['n_layer']
        self.modal_num = hyperpm['modal_num']
        self.n_class = hyperpm['num_classes']
        self.d_out = self.d_v * self.n_head * self.modal_num  # 24*4*2

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []

        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            # encoder = nn.MultiheadAttention(self.d_k * self.n_head, self.n_head, dropout = self.dropout) #nn.multi_head_attn
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer_Filter(d_in, self.d_v * self.n_head, self.n_class, self.modal_num, self.d_q,
                                              self.dropout)

    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())

        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            # x = x.transpose(1, 0)#nn.multi_head_attn
            # x, attn = self.Encoder[i](x, x, x)#nn.multi_head_attn
            # x = x.transpose(1, 0)#nn.multi_head_attn
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())

        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1)
        output, hidden = self.Outputlayer(x, attn_embedding)
        return output, hidden, attn_map


class VLTransformer_FilterLayer1(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(VLTransformer_FilterLayer1, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm['n_hidden']
        self.d_k = hyperpm['n_hidden']
        self.d_v = hyperpm['n_hidden']
        self.n_head = hyperpm['n_head']
        self.dropout = hyperpm['fusion_dropout']
        self.n_layer = hyperpm['n_layer']
        self.modal_num = hyperpm['modal_num']
        self.n_class = hyperpm['num_classes']
        self.d_out = self.d_v * self.n_head * self.modal_num  # 24*4*2

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []
        # seq_len, hidden_size, prob
        self.filter = FilterLayer(seq_len=300, hidden_size=3, prob=self.dropout)
        self.w_filter = nn.Linear(900, 900)

        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            # encoder = nn.MultiheadAttention(self.d_k * self.n_head, self.n_head, dropout = self.dropout) #nn.multi_head_attn
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer(d_in, self.d_v * self.n_head, self.n_class, self.modal_num, self.dropout)

    def forward(self, x):
        bs, hid = x.shape
        attn_map = []
        x = self.w_filter(x).view(bs, self.modal_num, hid // self.modal_num)
        # [batch, seq_len, hidden]
        x = x.transpose(1, 2)
        x = self.filter(x)
        x = x.transpose(1, 2)
        x = x.view(bs, -1)
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())

        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            # x = x.transpose(1, 0)#nn.multi_head_attn
            # x, attn = self.Encoder[i](x, x, x)#nn.multi_head_attn
            # x = x.transpose(1, 0)#nn.multi_head_attn
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())

        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1)
        output, hidden = self.Outputlayer(x, attn_embedding)
        return output, hidden, attn_map


class VLTransformer_Gate(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(VLTransformer_Gate, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm['n_hidden']
        self.d_k = hyperpm['n_hidden']
        self.d_v = hyperpm['n_hidden']
        self.n_head = hyperpm['n_head']
        self.dropout = hyperpm['fusion_dropout']
        self.n_layer = hyperpm['n_layer']
        self.modal_num = hyperpm['modal_num']
        self.n_class = hyperpm['num_classes']
        self.d_out = self.d_v * self.n_head * self.modal_num

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []

        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)

        self.FGLayer = FusionGate(self.modal_num)
        self.Outputlayer = OutputLayer(self.d_v * self.n_head, self.d_v * self.n_head, self.n_class, self.modal_num,
                                       self.dropout)

    def forward(self, x):
        bs = x.size(0)
        x, attn = self.InputLayer(x)
        attn_map = []
        attn = attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())
        for i in range(self.n_layer):
            x, attn_ = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            x = self.FeedForward[i](x)
            attn = attn_.mean(dim=1)
            attn_map.append(attn.detach().cpu().numpy())
        x, norm = self.FGLayer(x)
        x = x.sum(-2) / norm
        attn_embedding = attn.view(bs, -1)
        output, hidden = self.Outputlayer(x, attn_embedding)
        return output, hidden


class FusionGate(nn.Module):
    def __init__(self, channel, reduction=1):
        super(FusionGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x), y.sum(-2)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.wtrans = nn.Parameter(torch.zeros(size=(2 * out_features, out_features)))
        nn.init.xavier_uniform_(self.wtrans.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        h = torch.mm(inp, self.W)
        N = h.size()[0]
        Wh1 = torch.mm(h, self.a[:self.out_features, :])
        Wh2 = torch.mm(h, self.a[self.out_features:, :])
        e = self.leakyrelu(Wh1 + Wh2.T)
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        negative_attention = torch.where(adj > 0, -e, zero_vec)
        attention = F.softmax(attention, dim=1)
        negative_attention = -F.softmax(negative_attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        negative_attention = F.dropout(negative_attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, inp)
        h_prime_negative = torch.matmul(negative_attention, inp)
        h_prime_double = torch.cat([h_prime, h_prime_negative], dim=1)
        new_h_prime = torch.mm(h_prime_double, self.wtrans)

        if self.concat:
            return F.elu(new_h_prime)
        else:
            return new_h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Signed_GAT(nn.Module):
    def __init__(self, node_embedding, cosmatrix, nfeat, uV, original_adj, hidden=16, \
                 nb_heads=4, n_output=300, dropout=0, alpha=0.3):
        super(Signed_GAT, self).__init__()
        self.dropout = dropout
        self.uV = uV
        embedding_dim = 300
        self.user_tweet_embedding = nn.Embedding(num_embeddings=self.uV, embedding_dim=embedding_dim, padding_idx=0)
        self.user_tweet_embedding.from_pretrained(torch.from_numpy(node_embedding))
        self.original_adj = torch.from_numpy(original_adj.astype(np.float64)).cuda()
        self.potentinal_adj = torch.where(cosmatrix > 0.5, torch.ones_like(cosmatrix),
                                          torch.zeros_like(cosmatrix)).cuda()
        self.adj = self.original_adj + self.potentinal_adj
        self.adj = torch.where(self.adj > 0, torch.ones_like(self.adj), torch.zeros_like(self.adj))
        self.attentions = [GraphAttentionLayer(nfeat, n_output, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nb_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nfeat * nb_heads, n_output, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, X_tid):
        X = self.user_tweet_embedding(torch.arange(0, self.uV).long().cuda()).to(torch.float32)
        x = F.dropout(X, self.dropout, training=self.training)
        adj = self.adj.to(torch.float32)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.sigmoid(self.out_att(x, adj))
        return x[X_tid]


class Signed_GCN(nn.Module):
    def __init__(self, num_features, out_features, node_embedding, uV, original_adj, hidden=150, \
                dropout=0):
        super(Signed_GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, out_features)
        self.dropout = dropout
        self.uV = uV
        embedding_dim = 300
        self.user_tweet_embedding = nn.Embedding(num_embeddings=self.uV, embedding_dim=embedding_dim, padding_idx=0)
        self.user_tweet_embedding.from_pretrained(torch.from_numpy(node_embedding))
        self.adj = torch.from_numpy(original_adj.astype(np.int64)).cuda()

    def forward(self, X_tid):
        X = self.user_tweet_embedding(torch.arange(0, self.uV).long().cuda()).to(torch.float32)
        x = F.dropout(X, self.dropout, training=self.training)
        adj = self.adj
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)
        return x[X_tid]


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class FilterLayer(nn.Module):
    def __init__(self, seq_len, hidden_size, prob):
        super(FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, seq_len // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(p=prob)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerBlock(nn.Module):

    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v * n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, episilon=1e-6):
        '''
        :param Q: (*, max_q_words, n_heads, input_size)
        :param K: (*, max_k_words, n_heads, input_size)
        :param V: (*, max_v_words, n_heads, input_size)
        :param episilon:
        :return:
        '''
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)
        return V_att

    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads * self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))
        return output

    def forward(self, Q, K, V):
        '''
        :param Q: (batch_size, max_q_words, input_size)
        :param K: (batch_size, max_k_words, input_size)
        :param V: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        V_att = self.multi_head_attention(Q, K, V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output


class FaceAttributeDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FaceAttributeDecoder, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, mask):
        x = self.linear(x)
        return x


class GCN(nn.Module):
    def __init__(self, num_features, hidden, out_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, out_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# AFT
class hard_fc(nn.Module):
    def __init__(self, d_in, d_hid, DroPout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(DroPout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

ACT2FN = {'elu': F.elu, 'relu': F.relu, 'sigmoid': torch.sigmoid, 'tanh': torch.tanh}
class CorNetBlock(nn.Module):
    def __init__(self, context_size, output_size, cornet_act='sigmoid', **kwargs):
        super(CorNetBlock, self).__init__()
        self.dstbn2cntxt = nn.Linear(output_size, context_size)
        self.cntxt2dstbn = nn.Linear(context_size, output_size)
        self.act_fn = ACT2FN[cornet_act]

    def forward(self, output_dstrbtn):
        identity_logits = output_dstrbtn
        output_dstrbtn = self.act_fn(output_dstrbtn)
        context_vector = self.dstbn2cntxt(output_dstrbtn)
        context_vector = F.elu(context_vector)
        output_dstrbtn = self.cntxt2dstbn(context_vector)
        output_dstrbtn = output_dstrbtn + identity_logits
        return output_dstrbtn


class CorNet(nn.Module):
    def __init__(self, output_size, cornet_dim=100, n_cornet_blocks=2, **kwargs):
        super(CorNet, self).__init__()
        self.intlv_layers = nn.ModuleList(
            [CorNetBlock(cornet_dim, output_size, **kwargs) for _ in range(n_cornet_blocks)])
        for layer in self.intlv_layers:
            nn.init.xavier_uniform_(layer.dstbn2cntxt.weight)
            nn.init.xavier_uniform_(layer.cntxt2dstbn.weight)

    def forward(self, logits):
        for layer in self.intlv_layers:
            logits = layer(logits)
        return logits


# 对抗性训练
class PGD(object):

    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        # x 的形状应该是 (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # 返回 LSTM 的输出和最后一个隐藏状态
        return out, _


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


import pywt
import pywt.data
from functools import partial


# Wavelet Transform Conv(WTConv2d)
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


# 论文地址 https://arxiv.org/pdf/2407.05848
class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConvWithWTConv2d, self).__init__()

        # 深度卷积：使用 WTConv2d 替换 3x3 卷积
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size)

        # 逐点卷积：使用 1x1 卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# 论文地址：https://arxiv.org/abs/2305.13563v2
class EMA(nn.Module):
    def __init__(self, channels, factor=3):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


def batch_min_max_normalize(tensor_batch):
    min_vals = torch.min(tensor_batch, dim=0)[0]
    max_vals = torch.max(tensor_batch, dim=0)[0]
    normalized_batch = (tensor_batch - min_vals) / (max_vals - min_vals)
    return normalized_batch


class MLP_trans(torch.nn.Module):
    def __init__(self, input_size, out_size, dropout=0.2):
        super(MLP_trans, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.SiLU()
        self.mlp_1 = nn.Linear(input_size, out_size)
        self.mlp_2 = nn.Linear(out_size, out_size)
        # self.mlp_3 = nn.Linear(input_size, input_size)
        # self.mlp_4 = nn.Linear(input_size, out_size)

    def forward(self, emb_trans):
        emb_trans = self.dropout(self.activate(self.mlp_1(emb_trans)))
        emb_trans = self.dropout(self.activate(self.mlp_2(emb_trans)))
        # emb_trans = self.dropout(self.activate(self.mlp_3(emb_trans)))
        # emb_trans = self.dropout(self.activate(self.mlp_4(emb_trans)))
        return emb_trans


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, activate="relu", head_num=2, dropout=0, initializer_range=0.02):
        super(MultiHeadSelfAttention, self).__init__()
        self.config = list()

        self.hidden_size = hidden_size

        self.head_num = head_num
        if (self.hidden_size) % head_num != 0:
            raise ValueError(self.head_num, "error")
        self.head_dim = self.hidden_size // self.head_num

        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.concat_weight = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        torch.nn.init.normal_(self.query.weight, 0, initializer_range)
        torch.nn.init.normal_(self.key.weight, 0, initializer_range)
        torch.nn.init.normal_(self.value.weight, 0, initializer_range)
        torch.nn.init.normal_(self.concat_weight.weight, 0, initializer_range)
        self.dropout = torch.nn.Dropout(dropout)

    def dot_score(self, encoder_output):
        query = self.dropout(self.query(encoder_output))
        key = self.dropout(self.key(encoder_output))
        # head_num * batch_size * session_length * head_dim
        querys = torch.stack(query.chunk(self.head_num, -1), 0)
        keys = torch.stack(key.chunk(self.head_num, -1), 0)
        # head_num * batch_size * session_length * session_length
        dots = querys.matmul(keys.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        #         print(len(dots),dots[0].shape)
        return dots

    def forward(self, encoder_outputs, mask=None):
        attention_energies = self.dot_score(encoder_outputs)
        value = self.dropout(self.value(encoder_outputs))

        values = torch.stack(value.chunk(self.head_num, -1))

        if mask is not None:
            eye = torch.eye(mask.shape[-1]).to('cuda')
            new_mask = torch.clamp_max((1 - (1 - mask.float()).unsqueeze(1).permute(0, 2, 1).bmm(
                (1 - mask.float()).unsqueeze(1))) + eye, 1)
            attention_energies = attention_energies - new_mask * 1e12
            weights = F.softmax(attention_energies, dim=-1)
            weights = weights * (1 - new_mask)
        else:
            weights = F.softmax(attention_energies, dim=2)

        # head_num * batch_size * session_length * head_dim
        outputs = weights.matmul(values)
        # batch_size * session_length * hidden_size
        outputs = torch.cat([outputs[i] for i in range(outputs.shape[0])], dim=-1)
        outputs = self.dropout(self.concat_weight(outputs))

        return outputs


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_size, initializer_range=0.02):
        super(PositionWiseFeedForward, self).__init__()
        self.final1 = torch.nn.Linear(hidden_size, hidden_size * 4, bias=True)
        self.final2 = torch.nn.Linear(hidden_size * 4, hidden_size, bias=True)
        torch.nn.init.normal_(self.final1.weight, 0, initializer_range)
        torch.nn.init.normal_(self.final2.weight, 0, initializer_range)

    def forward(self, x):
        x = F.relu(self.final1(x))
        x = self.final2(x)
        return x


class TransformerLayer(torch.nn.Module):
    def __init__(self, hidden_size, activate="relu", head_num=4, dropout=0, attention_dropout=0,
                 initializer_range=0.02):
        super(TransformerLayer, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.mh = MultiHeadSelfAttention(hidden_size=hidden_size, activate=activate, head_num=head_num,
                                         dropout=attention_dropout, initializer_range=initializer_range)
        self.pffn = PositionWiseFeedForward(hidden_size, initializer_range=initializer_range)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, encoder_outputs, mask=None):
        encoder_outputs = self.layer_norm(encoder_outputs + self.dropout(self.mh(encoder_outputs, mask)))
        encoder_outputs = self.layer_norm(encoder_outputs + self.dropout(self.pffn(encoder_outputs)))
        return encoder_outputs