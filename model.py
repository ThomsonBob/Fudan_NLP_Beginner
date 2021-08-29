import torch
from torch import nn
import numpy as np
from transformers import BertModel
import fastNLP
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import dgl
import logging
import random

# from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h, g):
        z = self.fc(h)
        g.ndata['z'] = z  # equation (2)
        g.apply_edges(self.edge_attention)  # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge
        self.ffn = PositionwiseFeedForward(num_heads * out_dim, 2048, 0.1)

    def forward(self, h, g):
        head_outs = [attn_head(h, g) for attn_head in self.heads]
        h = F.elu(torch.cat(head_outs, dim=1)) + h
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        return h


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads)

    def forward(self, h, g):
        h = self.layer1(h, g)
        return h


def get_pair(label, len_review):
    pair = {}  # target = []  # the ground truth of the sampled reply node
    for i in range(len_review):
        if label[i]:
            if label[i] in pair:
                pair[label[i]].append(i)
            else:
                pair[label[i]] = [i]
    return pair


class APE(nn.Module):
    def __init__(self, config):
        super(APE, self).__init__()
        self.device = config.device
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.gat = GAT(in_dim=768, hidden_dim=256, num_heads=3)  # num_heads参数 可调
        self.emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(175 + 86, 768), freeze=True)
        self.pos = self.get_pos().to(self.device)
        self.encode_lstm = nn.LSTM(input_size=768, hidden_size=int(768 / 2), batch_first=True, bidirectional=True).to(
            self.device)
        self.decode_lstm = nn.LSTM(input_size=768, hidden_size=int(768 / 2), batch_first=True, bidirectional=True,
                                   num_layers=2).to(self.device)
        self.decode_linear = nn.Linear(768, 3, bias=True)
        self.crf = fastNLP.modules.ConditionalRandomField(3).to(self.device)  # number of tags
        # 不一定对
        # self.loss_function = torch.nn.BCELoss()
        self.biaffine = torch.nn.Bilinear(768, 768, 1).to(self.device)  # num_label = 2
        self.training = True
        self.reset_parameters()

    def get_pos(self):
        pos = []
        for i in range(175 + 86):
            pos.append(self.emb(torch.tensor(i)))
        return torch.stack(pos)

    def reset_parameters(self):
        self.biaffine.reset_parameters()
        # self.encode_lstm.reset_parameters()
        # self.decode_linear.reset_parameters()
        # self.decode_lstm.reset_parameters()

    def build_graph(self, cls, len_review, len_reply, label=None, metric=None):
        features = cls[:len_review + len_reply]
        nodes = [i for i in range(len_review + len_reply - 1)]
        edges = [i + 1 for i in range(len_review + len_reply - 1)]
        G = dgl.graph((nodes + edges, nodes + edges)).to(self.device)
        b = []  # begin node of an edge
        e = []  # end node of an edge
        if label is not None:
            pair = {}  # target = []  # the ground truth of the sampled reply node
            for i in range(len_review):
                if label[i]:
                    if label[i] in pair:
                        pair[label[i]].append(i)
                    else:
                        pair[label[i]] = [i]
            for i in range(len_review, len_review + len_reply):
                if label[i] and label[i] in pair:
                    b += [i] * len(pair[label[i]])
                    e += pair[label[i]]
        elif metric is not None:
            for i in range(len_review * len_reply):
                if metric[i]:
                    b.append(i % len_review)
                    e.append(len_review + i // len_review)
        G.add_edges(b + e, e + b)  # bidirectional edges
        return G, features

    def build_batch(self, cls, len_review, len_reply, label_pairs=None, metric=None):
        g = []  # graph of a batch
        f = []  # features of the nodes
        if label_pairs is not None:
            for i in range(len(len_review)):
                j, k = self.build_graph(cls[i], len_review[i], len_reply[i], label=label_pairs[i])
                g.append(j)
                f.append(k)
        elif metric is not None:
            for i in range(len(len_review)):
                j, k = self.build_graph(cls[i], len_review[i], len_reply[i], metric=metric[i])
                g.append(j)
                f.append(k)
        return g, torch.cat(f, dim=0)

    def forward(self, Data):
        """
        === dimension of each parameter ===
        tokens: (num of passages) * (num of sentences) * 512
        labels / label_pairs: (num of passages) * (num of sentences) note that labels is tensor, not int
        cls: (num of passages) * (num of max_sent) * 768
        len_review / len_reply: num of passages
        """
        tokens, labels, cls, len_review, len_reply, label_pairs = Data
        if self.training:  # .train()
            bert_id = [random.choice(list(range(i + j))) for i, j in zip(len_review, len_reply)]
            for i, j in enumerate(bert_id):
                s = self.bert(tokens[i][j].to(self.device))[1][0]
                cls[i][j] = s
        cls = torch.stack([i.to(self.device) + self.pos[:i.shape[0]] for j, i in enumerate(cls)])
        data = pack_padded_sequence(cls, [len_reply[i] + len_review[i] for i in range(len(cls))], batch_first=True)
        data, _ = self.encode_lstm(data)
        data, _ = pad_packed_sequence(data)
        data = data.transpose(0, 1)
        loss = 0.0
        pair_truth = torch.zeros((len(len_review), max(len_review) * max(len_reply)), dtype=torch.bool)
        overall_metric = torch.zeros((len(len_review), max(len_review) * max(len_reply)), dtype=torch.bool)
        if self.training:
            for i in range(len(len_review)):
                all_label = label_pairs[i]
                nodes = [i for i in range(len_review[i] + len_reply[i] - 1)]
                edges = [i + 1 for i in range(len_review[i] + len_reply[i] - 1)]
                pair = get_pair(all_label, len_review[i])
                g = dgl.graph((nodes + edges, nodes + edges)).to(self.device)
                h = data[i][:len(all_label)]
                for j in range(len_review[i], len_review[i] + len_reply[i]):
                    target = torch.zeros(len_review[i], dtype=torch.float)
                    gat_data = self.gat(h, g)
                    reply = gat_data[j].unsqueeze(0).repeat(len_review[i], 1)
                    pair_logits = self.biaffine(gat_data[:len_review[i]], reply)
                    pred_logits = torch.sigmoid(pair_logits).squeeze(1)
                    if all_label[j] != 0:
                        # logger.info(pred_logits)
                        target = torch.eq(torch.tensor(all_label[:len_review[i]]), all_label[j]).float()
                    # beta = torch.sum(target) / target.size(0)
                    class_weights = target.clone().detach()
                    class_weights = class_weights + (1 - class_weights)/target.size(0)
                    class_weights = class_weights.to(self.device)
                    loss_function = torch.nn.BCELoss(weight=class_weights)
                    target = target.to(self.device)
                    # logger.info(target)
                    loss += loss_function(pred_logits, target) / (len_reply[i] * len_review[i])
                    if all_label[j] and all_label[j] in pair:
                        g.add_edges([j] * len(pair[all_label[j]]) + pair[all_label[j]],
                                    pair[all_label[j]] + [j] * len(pair[all_label[j]]))
                        # logger.info([j] * len(pair[all_label[j]]) + pair[all_label[j]])
            G, H = self.build_batch(data, len_review, len_reply, label_pairs=label_pairs)
            G = dgl.batch(G)
            data = self.gat(H, G)
            l = [0] * (1 + len(len_review))  # 拆分batch
            for i in range(len(len_review)):
                l[i + 1] = len_review[i] + len_reply[i] + l[i]
            data = [data[l[i - 1]:l[i]] for i in range(1, len(l))]
            pad = torch.full((1, 768), 0).to(self.device)  # sentence
            data = torch.stack(
                [torch.cat([data] + [pad] * int(len_review[0] + len_reply[0] - data.shape[0]), dim=0) for data in data])
            data = pack_padded_sequence(data, [len_reply[i] + len_review[i] for i in range(len(cls))], batch_first=True)
            data, _ = self.decode_lstm(data)
            data, _ = pad_packed_sequence(data)
            data = data.transpose(0, 1)
            data = self.decode_linear(data)  # 标签的得分 三分类
            return (self.crf.forward(data, labels, torch.BoolTensor(
                [[1] * (i + j) + [0] * (labels.shape[1] - i - j) for i, j in zip(len_review, len_reply)]).to(
                self.device)), loss)
        else:
            for i in range(len(len_review)):
                for j, reply in enumerate(label_pairs[i][len_review[i]:]):
                    for k, review in enumerate(label_pairs[i][:len_review[i]]):
                        if review and review == reply:
                            pair_truth[i][j * len_review[i] + k] = 1
            # count = 0
            for i in range(len(len_review)):
                # count += len_review[i]
                # count += len_reply[i]
                g, h = self.build_graph(data[i], len_review[i], len_reply[i])
                temp_data = self.gat(h, g)
                for j in range(len_reply[i]):
                    reply = temp_data[j + len_review[i]].unsqueeze(0).repeat(len_review[i], 1)
                    pair_logits = self.biaffine(temp_data[:len_review[i]], reply)
                    pred_logits = torch.sigmoid(pair_logits).squeeze(1)
                    label = torch.ge(pred_logits, 0.5)
                    overall_metric[i][j * len_review[i]:(j + 1) * len_review[i]] = label
                    # logger.info(label)
                    b = torch.nonzero(label).view(-1).tolist()
                    e = [j + len_review[i]] * len(b)
                    g.add_edges(b + e, e + b)
                    temp_data = self.gat(h, g)
            G, H = self.build_batch(data, len_review, len_reply, metric=overall_metric)
            G = dgl.batch(G)
            # logger.info(len(G.nodes()))
            # logger.info(f"count{count}")
            data = self.gat(H, G)
            l = [0] * (1 + len(len_review))  # 拆分batch
            for i in range(len(len_review)):
                l[i + 1] = len_review[i] + len_reply[i] + l[i]
            data = [data[l[i - 1]:l[i]] for i in range(1, len(l))]
            pad = torch.full((1, 768), 0).to(self.device)  # sentence
            data = torch.stack(
                [torch.cat([data] + [pad] * int(len_review[0] + len_reply[0] - data.shape[0]), dim=0) for data in data])
            data = pack_padded_sequence(data, [len_reply[i] + len_review[i] for i in range(len(cls))], batch_first=True)
            data, _ = self.decode_lstm(data)
            data, _ = pad_packed_sequence(data)
            data = data.transpose(0, 1)
            data = self.decode_linear(data)  # 标签的得分 三分类
            return (self.crf.viterbi_decode(
                data, torch.BoolTensor([[1] * (i + j) + [0] * (labels.shape[1] - i - j) for i, j in
                                        zip(len_review, len_reply)]).to(self.device))[0], overall_metric, pair_truth)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """
            sinusoid的embedding，其中position的表示中，偶数维(0,2,4,...)是sin, 奇数(1,3,5...)是cos
            :param int n_position: 一共多少个position
            :param int d_hid: 多少维度，需要为偶数
            :param padding_idx:
            :return: torch.FloatTensor, shape为n_position x d_hid
            """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)  # (175+86) * 768
