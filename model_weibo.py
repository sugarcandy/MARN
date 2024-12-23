"""
多模态融合+对比学习
其中对比学习用于将同标签的距离拉近，不同标签的距离拉远
使用对比学习进行模态对齐
此处的loss之间的权重为手动设置

在best的基础上去掉梯度裁剪
"""

import os

# from ClippyAdam import ClippyAdam

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import abc
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.init as init
import pickle
import json, os, time
import argparse
import warnings
import random
import sys
import clip
from PIL import Image

sys.path.append('/../image_part')
from graph_part.code_v1 import config_file
from graph_part.layer import *
from graph_part.utils import *
from graph_part.align_loss import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.description = "ini"
parser.add_argument("-t", "--task", type=str, default="weibo2")
parser.add_argument("-g", "--gpu_id", type=str, default="1")
parser.add_argument("-c", "--config_name", type=str, default="single3.json")
parser.add_argument("-T", "--thread_name", type=str, default="Thread-1")
parser.add_argument("-d", "--description", type=str, default="exp_description")
args = parser.parse_args()


def process_config(config):
    for k, v in config.items():
        config[k] = v[0]
    return config


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.init_clip_max_norm = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abc.abstractmethod
    def forward(self):
        pass

    def mfan(self, x_tid, x_text, y, loss, i, total, params, pgd_word):
        self.optimizer.zero_grad()
        # 这里调用mfan的forward
        logit_original, dist_og, hidden, features = self.forward(x_tid, x_text)
        # print(f"第{i}个批次的logits")
        # print(logit_original)
        # print(f"第{i}个批次的label")
        # print(y)
        # 分类损失
        loss_classification = loss(logit_original, y)

        # 模态对齐损失
        loss_mse = nn.MSELoss()
        loss_dis = loss_mse(dist_og[0], dist_og[1])

        # hidden的shape为[batch_size,output_dim]
        # 此时features的shape为[batch_size,2,output_dim],为每一份数据创建了一个副本，进行对比
        loss_cons = SupConLoss(temperature=2.0) ## 1.0时候0.8949
        loss_constrative = loss_cons(features, y)

        losses = [loss_classification, loss_constrative, loss_dis]
        important = [loss_classification]
        loss_defense = geometric_loss(losses, important)
        # loss_defense=detach_loss_v2(losses)
        loss_defense.backward()

        # 对抗性训练
        K = 3
        pgd_word.backup_grad()
        for t in range(K):
            pgd_word.attack(is_first_attack=(t == 0))
            if t != K - 1:
                self.zero_grad()
            else:
                pgd_word.restore_grad()
            loss_adv, dist, _, _ = self.forward(x_tid, x_text)
            loss_adv = loss(loss_adv, y)
            loss_adv.backward()
        pgd_word.restore()
        # nn.utils.clip_grad_norm_(self.parameters(), max_norm=10, norm_type=2)  # 使用第二种裁剪方式
        self.optimizer.step()
        corrects = (torch.max(logit_original, 1)[1].view(y.size()).data == y.data).sum()
        accuracy = 100 * corrects / len(y)
        print(
            'Batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                           loss_defense.item(),
                                                                           accuracy,
                                                                           corrects,
                                                                           y.size(0)))

    def fit(self, X_train_tid, X_train, y_train,
            X_dev_tid, X_dev, y_dev):

        if torch.cuda.is_available():
            self.cuda()
        batch_size = self.config['batch_size']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)  # 原始值为2e-3

        X_train_tid = torch.LongTensor(X_train_tid)
        X_train = torch.LongTensor(X_train)
        y_train = torch.LongTensor(y_train)

        # DataLoader
        dataset = TensorDataset(X_train_tid, X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        loss = nn.NLLLoss()
        params = [(name, param) for name, param in self.named_parameters()]
        pgd_word = PGD(self, emb_name='word_embedding', epsilon=6, alpha=1.8)

        for epoch in range(self.config['epochs']):
            print("\nEpoch ", epoch + 1, "/", self.config['epochs'])
            self.train()
            avg_loss = 0
            avg_acc = 0
            for i, data in enumerate(dataloader):
                total = len(dataloader)
                batch_x_tid, batch_x_text, batch_y = (item.cuda(device=self.device) for item in data)
                self.mfan(batch_x_tid, batch_x_text, batch_y, loss, i, total, params, pgd_word)
            self.evaluate(X_dev_tid, X_dev, y_dev)

    def evaluate(self, X_dev_tid, X_dev, y_dev):
        y_pred = self.predict(X_dev_tid, X_dev)
        acc = accuracy_score(y_dev, y_pred)

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.state_dict(), self.config['save_path'])
            print(classification_report(y_dev, y_pred, target_names=self.config['target_names'], digits=5))
            print("Val set acc:", acc)
            print("Best val set acc:", self.best_acc)
            print("saved model at ", self.config['save_path'])

    def predict(self, X_test_tid, X_test):
        if torch.cuda.is_available():
            self.cuda()
        self.eval()
        y_pred = []
        X_test_tid = torch.LongTensor(X_test_tid).cuda()
        X_test = torch.LongTensor(X_test).cuda()

        dataset = TensorDataset(X_test_tid, X_test)
        dataloader = DataLoader(dataset, batch_size=50)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_tid, batch_x_text = (item.cuda(device=self.device) for item in data)
                logits, dist, _, _ = self.forward(batch_x_tid, batch_x_text)
                predicted = torch.max(logits, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred


## 通过Clip加载图像
class ClipModel:
    def __init__(self, model, preprocess, device):
        self.newid2imgnum = config['newid2imgnum']
        self.model = model
        self.model.visual.attnpool.c_proj = torch.nn.Linear(2048, 300, dtype=torch.float16).cuda()
        torch.nn.init.eye_(self.model.visual.attnpool.c_proj.weight)
        self.path = os.path.dirname(os.getcwd()) + '/../dataset/weibo/weibo_images/weibo_images_all/'
        self.trans = preprocess
        self.device = device

    def forward(self, xtid):
        img_list = []
        for newid in xtid.cpu().numpy():
            imgnum = self.newid2imgnum[newid]
            imgpath = self.path + imgnum + '.jpg'
            im = self.trans(Image.open(imgpath)).unsqueeze(0)
            img_list.append(im)
        batch_img = torch.cat(img_list, dim=0).to(device=self.device)
        with torch.no_grad():
            img_output = self.model.encode_image(batch_img)  ## (64,512)
        return img_output.to(torch.float32)


class MARN(NeuralNetwork):
    def __init__(self, config, adj, original_adj):
        super(MARN, self).__init__()
        self.config = config
        self.uV = adj.shape[0]
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        self.mh_attention = TransformerBlock(input_size=300, n_heads=8, attn_dropout=0)
        self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
                                           _weight=torch.from_numpy(embedding_weights))
        self.text_embedding = LSTMEncoder(input_size=300, hidden_size=300, num_layers=1)
        self.cosmatrix = self.calculate_cos_matrix()
        self.gat_relation = Signed_GAT(node_embedding=config['node_embedding'], cosmatrix=self.cosmatrix, nfeat=300, \
                                       uV=self.uV, nb_heads=1,
                                       original_adj=original_adj, dropout=0)
        self.image_embedding = resnet50()
        clip_model, preprocess = clip.load("RN50", device=self.device)  ## ViT-B/32效果不好,这个0.8831
        self.image_embedding2 = ClipModel(clip_model, preprocess, device=self.device)
        fusion_dim = config['fusion_dim']
        self.model_fusion = VLTransformer(fusion_dim, config)
        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])
        self.relu = nn.ReLU()
        self.alignfc_g = nn.Linear(in_features=300, out_features=300)
        self.alignfc_t = nn.Linear(in_features=300, out_features=300)
        self.init_weight()

    def calculate_cos_matrix(self):
        a, b = torch.from_numpy(config['node_embedding']), torch.from_numpy(config['node_embedding'].T)
        c = torch.mm(a, b)
        aa = torch.mul(a, a)
        bb = torch.mul(b, b)
        asum = torch.sqrt(torch.sum(aa, dim=1, keepdim=True))
        bsum = torch.sqrt(torch.sum(bb, dim=0, keepdim=True))
        norm = torch.mm(asum, bsum)
        res = torch.div(c, norm)
        return res

    def init_weight(self):
        self.model_fusion.apply(my_weight_init)

    def forward(self, X_tid, X_text):
        # 获取文本嵌入
        X_text = self.word_embedding(X_text)
        if self.config['user_self_attention'] == True:
            X_text = self.mh_attention(X_text, X_text, X_text)
        # 修改文本编码器为lstm
        X_text, _ = self.text_embedding(X_text)
        X_text = X_text.permute(0, 2, 1)
        # 从GAT获取传播结构
        rembedding = self.gat_relation.forward(X_tid)
        # # resnet50获取图像嵌入
        # iembedding = self.image_embedding.forward(X_tid)
        ## 更换图像编码器为Cliptengx
        iembedding = self.image_embedding2.forward(X_tid)
        conv_block = [rembedding]
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(X_text))
            pool = max_pooling(act)
            pool = torch.squeeze(pool, dim=-1)
            conv_block.append(pool)

        conv_feature = torch.cat(conv_block, dim=1)
        graph_feature, text_feature = conv_feature[:, :300], conv_feature[:, 300:]
        bsz = text_feature.size()[0]

        # text_feature->[bach_size,300],graph_feature->[bach_size,300],iembedding->[batch_size,300]
        fusion_input = torch.concat([text_feature, iembedding, graph_feature], dim=-1)
        prob, hidden, attn = self.model_fusion(fusion_input)

        self_att_t = self.mh_attention(text_feature.view(bsz, -1, 300), text_feature.view(bsz, -1, 300), \
                                       text_feature.view(bsz, -1, 300))
        self_att_g = self.mh_attention(graph_feature.view(bsz, -1, 300), graph_feature.view(bsz, -1, 300), \
                                       graph_feature.view(bsz, -1, 300))
        self_att_i = self.mh_attention(iembedding.view(bsz, -1, 300), iembedding.view(bsz, -1, 300), \
                                       iembedding.view(bsz, -1, 300))
        text_enhanced = self.mh_attention(self_att_i.view((bsz, -1, 300)), self_att_t.view((bsz, -1, 300)), \
                                          self_att_t.view((bsz, -1, 300))).view(bsz, 300)

        # 特征对齐
        align_text = self.alignfc_t(text_enhanced)
        align_rembedding = self.alignfc_g(self_att_g)
        dist = [align_text, align_rembedding]

        features = torch.concat([hidden.unsqueeze(1), hidden.unsqueeze(1)], dim=1)
        return prob, dist, hidden, features


def load_dataset():
    pre = os.path.dirname(os.getcwd()) + '/../dataset/weibo/weibo_files'
    X_train_tid, X_train, y_train, word_embeddings, adj = pickle.load(open(pre + "/train.pkl", 'rb'))
    X_dev_tid, X_dev, y_dev = pickle.load(open(pre + "/dev.pkl", 'rb'))
    X_test_tid, X_test, y_test = pickle.load(open(pre + "/test.pkl", 'rb'))
    config['embedding_weights'] = word_embeddings
    config['node_embedding'] = pickle.load(open(pre + "/node_embedding.pkl", 'rb'))[0]
    print("#nodes: ", adj.shape[0])

    # newid2mid处理成类似格式：{1: 'zhangsan', 2: 'lisi', 3: 'wangwu', 4: 'zhaoliu', 5: 'zhouqi'}，新闻id与结点标对应
    with open(pre + '/new_id_dic.json', 'r') as f:
        newid2mid = json.load(f)
        newid2mid = dict(zip(newid2mid.values(), newid2mid.keys()))
    mid2num = {}
    # 是类似格式：{'z5qFIwiEj': 1, 'yBpiLiBnk': 2,.....} ，结点下标与新闻id
    for file in os.listdir(
            os.path.dirname(os.getcwd()) + '/../dataset/weibo/weibocontentwithimage/original-microblog/'):
        mid2num[file.split('_')[-2]] = file.split('_')[0]

    newid2num = {}
    for id in X_train_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_dev_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_test_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    config['newid2imgnum'] = newid2num

    return X_train_tid, X_train, y_train, \
           X_dev_tid, X_dev, y_dev, \
           X_test_tid, X_test, y_test, adj


def load_original_adj(adj):
    pre = os.path.dirname(os.getcwd()) + '/../dataset/weibo/weibo_files/'
    path = os.path.join(pre, 'original_adj')
    with open(path, 'r') as f:
        original_adj_dict = json.load(f)
    original_adj = np.zeros(shape=adj.shape)
    for i, v in original_adj_dict.items():
        v = [int(e) for e in v]
        original_adj[int(i), v] = 1
    return original_adj


def train_and_test(model):
    model_suffix = "mfan"
    res_dir = 'exp_result'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.task)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.description)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = config['save_path'] = os.path.join(res_dir, 'best_model_in_each_config')
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    # 最优模型路径
    config['save_path'] = os.path.join(res_dir, args.thread_name + '_' + 'config' + args.config_name.split(".")[
        0] + '_best_model_weights_' + model_suffix + 'lstmwithclip_nllLoss_modify')
    dir_path = os.path.join('exp_result', args.task, args.description)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    #
    # if os.path.exists(config['save_path']):
    #     os.system('rm {}'.format(config['save_path']))

    X_train_tid, X_train, y_train, \
    X_dev_tid, X_dev, y_dev, \
    X_test_tid, X_test, y_test, adj = load_dataset()
    original_adj = load_original_adj(adj)
    nn = model(config, adj, original_adj)
    # if torch.cuda.device_count() > 1:  ## 自定义的训练方法导致多卡训练无效
    #     print("多卡并行！")
    #     nn = torch.nn.DataParallel(model(config, adj, original_adj))
    #     nn = nn.module

    # nn.fit(X_train_tid, X_train, y_train,
    #        X_dev_tid, X_dev, y_dev)

    # # 当前最佳模型
    # nn.load_state_dict(torch.load(
    #     "exp_result/weibo2/exp_description/best_model_in_each_config/Thread-1_configsingle3_best_model_weights_mfanlstmwithclip_nllLoss_modify"))

    # 最佳模型
    nn.load_state_dict(torch.load(
        "exp_result/weibo2/exp_description/best_model_in_each_config/best_model"))
    y_pred = nn.predict(X_test_tid, X_test)
    res = classification_report(y_test, y_pred, target_names=config['target_names'], digits=3, output_dict=True)
    for k, v in res.items():
        print(k, v)
    print("result:{:.4f}".format(res['accuracy']))
    res2 = {}
    res_final = {}
    res_final.update(res)
    res_final.update(res2)
    print(res)
    return res

config = process_config(config_file.config)
seed = config['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
model = MARN
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# torch.use_deterministic_algorithms(True)  # 查找为啥不能复现代码的原因
train_and_test(model)
