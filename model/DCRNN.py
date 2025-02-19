import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import utils
import math

from model.cell import DCGRUCell

VERY_SMALL_NUMBER = 1e-12

class ConcreteGraphLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, num_supports=2, temperature=0.1, device='cpu'):
        super(ConcreteGraphLearner, self).__init__()
        self.num_nodes = num_nodes
        self.num_supports = num_supports
        self.temperature = temperature
        self.device = device
        # 为每个支持矩阵创建一个独立的MLP
        self.node_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_supports)
        ])
        # 可选：添加正则化层或其他机制以控制图的稀疏性

    def forward(self, x):
        # x: (B, T, N, d)
        # 对时间维求均值： (B, N, d)
        x_agg = x.mean(dim=1)  # (B, N, d)
        B = x_agg.size(0)

        A_list = []
        for head in range(self.num_supports):
            node_embed = self.node_mlps[head](x_agg)  # (B, N, hidden_dim)
            # 计算 π_u,v = sigmoid(Z(u) * Z(v)^T)
            sim = torch.matmul(node_embed, node_embed.transpose(-1, -2))  # (B, N, N)
            pi = torch.sigmoid(sim)  # (B, N, N)

            # 使用Concrete分布对Bernoulli(π)进行连续近似
            eps = torch.rand_like(pi)
            eps = torch.clamp(eps, min=VERY_SMALL_NUMBER, max=1 - VERY_SMALL_NUMBER)
            logit = (torch.log(pi + VERY_SMALL_NUMBER) - torch.log(1. - pi + VERY_SMALL_NUMBER) +
                     torch.log(eps + VERY_SMALL_NUMBER) - torch.log(1. - eps + VERY_SMALL_NUMBER))
            A_concrete = torch.sigmoid(logit / self.temperature)  # (B, N, N)

            A_list.append(A_concrete.unsqueeze(1))  # (B, 1, N, N)

        A_batch = torch.cat(A_list, dim=1)  # (B, K, N, N)
        return A_batch  # (B, K, N, N)

class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step, hid_dim, num_nodes,
                 num_rnn_layers, dcgru_activation=None, filter_type='laplacian', device=None):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers
        self.num_nodes = num_nodes
        self._device = device
        self.num_supports = max_diffusion_step  # K = max_diffusion_step

        cells = []
        # 第一层 DCGRUCell，input_dim=args.input_dim=128
        cells.append(
            DCGRUCell(input_dim=input_dim, num_units=hid_dim, max_diffusion_step=max_diffusion_step,
                      num_nodes=num_nodes, nonlinearity=dcgru_activation, filter_type=filter_type))
        # 后续层 DCGRUCell，input_dim=hid_dim=128
        for _ in range(1, num_rnn_layers):
            cells.append(
                DCGRUCell(input_dim=hid_dim, num_units=hid_dim, max_diffusion_step=max_diffusion_step,
                          num_nodes=num_nodes, nonlinearity=dcgru_activation, filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(cells)

    def forward(self, inputs, initial_hidden_state, supports):
        # inputs: (B, T, N, d) -> (T, B, N*d)
        seq_length = inputs.shape[1]
        batch_size = inputs.shape[0]
        inputs = inputs.permute(1, 0, 2, 3).contiguous().view(seq_length, batch_size, -1)  # (T, B, N*d)

        current_inputs = inputs
        output_hidden = []
        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]  # (B, num_units)
            output_inner = []
            for t in range(seq_length):
                input_t = current_inputs[t]  # (B, input_dim)
                _, hidden_state = self.encoding_cells[i_layer](supports[:, t, :, :], input_t, hidden_state)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).to(self._device)  # (T, B, num_units)
        output_hidden = torch.stack(output_hidden, dim=0).to(self._device)  # (num_layers, B, num_units)
        return output_hidden, current_inputs  # (num_layers, B, num_units), (T, B, num_units)

    def init_hidden(self, batch_size):
        init_states = []
        for i, cell in enumerate(self.encoding_cells):
            hidden = cell.init_hidden(batch_size)  # (B, num_units)
            init_states.append(hidden)
        return torch.stack(init_states, dim=0)  # (num_layers, B, num_units)

class DCRNNModel_classification(nn.Module):
    def __init__(self, args, num_classes, device=None):
        super(DCRNNModel_classification, self).__init__()
        self.args = args
        self.num_nodes = args.num_nodes
        self.num_classes = num_classes
        self._device = device

        self.encoder = DCRNNEncoder(
            input_dim=args.input_dim,
            max_diffusion_step=args.max_diffusion_step,
            hid_dim=args.rnn_units,
            num_nodes=args.num_nodes,
            num_rnn_layers=args.num_rnn_layers,
            dcgru_activation=args.dcgru_activation,
            filter_type=args.filter_type,
            device=device
        )

        # 使用ConcreteGraphLearner
        self.graph_learner = ConcreteGraphLearner(
            input_dim=args.input_dim,
            hidden_dim=args.rnn_units,  # 与args.hidden_dim保持一致
            num_nodes=args.num_nodes,
            num_supports=2,  # 与原supports数量一致
            temperature=0.1, 
            device=device
        )

        # 信息瓶颈维度
        self.IB_size = args.IB_size
        self.mu_layer = nn.Linear(args.rnn_units, self.IB_size)
        self.sigma_layer = nn.Linear(args.rnn_units, self.IB_size)

        self.classifier = nn.Linear(self.IB_size, self.num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def preprocess_adj(self, A):
        # A: (B, K, N, N)
        # 逐个支持矩阵进行归一化
        B, K, N, _ = A.size()
        A_norm = torch.zeros_like(A)
        for k in range(K):
            A_k = A[:, k, :, :]  # (B, N, N)
            d = torch.sum(A_k, dim=-1, keepdim=True)  # (B, N, 1)
            d_inv_sqrt = torch.pow(d + 1e-10, -0.5)
            A_norm_k = A_k * d_inv_sqrt * d_inv_sqrt.transpose(-1, -2)  # (B, N, N)
            A_norm[:, k, :, :] = A_norm_k
        return A_norm  # (B, K, N, N)

    def forward(self, x, seq_lengths, supports):
        # x: (B, T, N, d)
        A_IB = self.graph_learner(x)  # (B, K, N, N), continuous relaxation

        # 预处理A_IB以生成new_supports
        new_supports = self.preprocess_adj(A_IB)  # (B, K, N, N)

        # 如果原始supports有多个支持矩阵（如K=2），则需要调整new_supports的形状
        # 原始supports形状为 [B, T, K, N, N]
        # new_supports目前为 [B, K, N, N]
        # 需要扩展到 [B, T, K, N, N]
        # 通过在时间维度上重复K个支持矩阵
        T = supports.shape[1]
        new_supports = new_supports.unsqueeze(1).repeat(1, T, 1, 1, 1)  # (B, T, K, N, N)

        # 初始化隐藏状态
        init_hidden = self.encoder.init_hidden(x.size(0)).to(self._device)

        # 编码器前向传播
        encoder_hidden_state, final_hidden = self.encoder(x, init_hidden, new_supports)  # (num_layers, B, num_units), (T, B, num_units)

        # 转置以适应后续操作
        output = final_hidden.permute(1, 0, 2)  # (B, T, num_units)

        # 提取最后有效时间步的输出
        last_out = utils.last_relevant_pytorch(output, seq_lengths, batch_first=True)  # (B, num_units)

        # 将last_out从 (B, num_units) 转换为 (B, N, r) 以进行节点池化
        last_out = last_out.view(x.size(0), self.num_nodes, -1)  # (B, N, r)

        # 对节点进行池化（例如，max pooling）
        pooled = torch.max(last_out, dim=1)[0]  # (B, r)

        # 信息瓶颈：参数化高斯分布
        mu = self.mu_layer(pooled)  # (B, IB_size)
        sigma = F.softplus(self.sigma_layer(pooled))  # (B, IB_size)

        # 重参数化采样
        eps = torch.randn_like(sigma)
        Z = mu + sigma * eps  # (B, IB_size)

        # 分类器
        logits = self.classifier(self.relu(self.dropout(Z)))  # (B, num_classes)

        return logits, Z, mu, sigma

########## Model for next time prediction ##########
class DCRNNModel_nextTimePred(nn.Module):
    def __init__(self, args, device=None):
        super(DCRNNModel_nextTimePred, self).__init__()

        num_nodes = args.num_nodes
        num_rnn_layers = args.num_rnn_layers
        rnn_units = args.rnn_units
        enc_input_dim = args.input_dim
        dec_input_dim = args.output_dim
        output_dim = args.output_dim
        max_diffusion_step = args.max_diffusion_step

        self.num_nodes = args.num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.output_dim = output_dim
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = bool(args.use_curriculum_learning)

        self.encoder = DCRNNEncoder(input_dim=enc_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    hid_dim=rnn_units, num_nodes=num_nodes,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation=args.dcgru_activation,
                                    filter_type=args.filter_type,
                                    device=device)
        self.decoder = DCGRUDecoder(input_dim=dec_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    num_nodes=num_nodes, hid_dim=rnn_units,
                                    output_dim=output_dim,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation=args.dcgru_activation,
                                    filter_type=args.filter_type,
                                    device=device,
                                    dropout=args.dropout)

    def forward(
            self,
            encoder_inputs,
            decoder_inputs,
            supports,
            batches_seen=None):
        """
        Args:
            encoder_inputs: encoder input sequence, shape (batch, input_seq_len, num_nodes, input_dim)
            decoder_inputs: decoder input sequence, shape (batch, output_seq_len, num_nodes, output_dim)
            supports: list of supports from laplacian or dual_random_walk filters
            batches_seen: number of examples seen so far, for teacher forcing
        Returns:
            outputs: predicted output sequence, shape (batch, output_seq_len, num_nodes, output_dim)
        """
        batch_size, output_seq_len, num_nodes, _ = decoder_inputs.shape

        # (seq_len, batch_size, num_nodes, input_dim)
        encoder_inputs = torch.transpose(encoder_inputs, dim0=0, dim1=1)
        # (seq_len, batch_size, num_nodes, output_dim)
        decoder_inputs = torch.transpose(decoder_inputs, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(batch_size).to(self._device)

        # encoder
        # (num_layers, batch, N*rnn_units)
        encoder_hidden_state, final_hidden = self.encoder(
            encoder_inputs, init_hidden_state, supports)

        # decoder
        if self.training and self.use_curriculum_learning and (
                batches_seen is not None):
            teacher_forcing_ratio = utils.compute_sampling_threshold(
                self.cl_decay_steps, batches_seen)
        else:
            teacher_forcing_ratio = None
        outputs = self.decoder(
            decoder_inputs,
            encoder_hidden_state,
            supports,
            teacher_forcing_ratio=teacher_forcing_ratio)  # (seq_len, batch_size, num_nodes * output_dim)
        # (seq_len, batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((output_seq_len, batch_size, num_nodes, -1))
        # (batch_size, seq_len, num_nodes, output_dim)
        outputs = torch.transpose(outputs, dim0=0, dim1=1)

        return outputs
########## Model for next time prediction ##########
