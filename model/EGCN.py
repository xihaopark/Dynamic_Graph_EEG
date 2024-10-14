import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import utils
import numpy as np

class EGCN(torch.nn.Module):
    def __init__(self, args, activation, device='cpu', skipfeats=False):
        super().__init__()
        GRCU_args = u.Namespace({})

        feats = [args.input_dim,
                 args.input_dim,
                 args.input_dim]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = nn.ModuleList()  
        for i in range(1, len(feats)):
            GRCU_args = u.Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation})

            grcu_i = GRCU(GRCU_args)
            self.GRCU_layers.append(grcu_i.to(self.device))

    def forward(self, Nodes_list, A_list):
        # Sparse TensorをDense Tensorに変換
        dense_A_list = [A.to_dense() for A in A_list]
        
        # Dense TensorをCPUに移動しNumPy配列に変換
        np_A_list = [A.cpu().numpy() for A in dense_A_list]
        # NumPy配列の形状を表示
        # print("A_list", np_A_list[0])

        # Nodes_listも同様に処理（リストの各要素に対して処理を行う）
        if isinstance(Nodes_list, list):
            dense_Nodes_list = [N.to_dense() if N.is_sparse else N for N in Nodes_list]
            np_Nodes_list = [N.cpu().numpy() for N in dense_Nodes_list]
        else:
            dense_Nodes_list = Nodes_list.to_dense() if Nodes_list.is_sparse else Nodes_list
            np_Nodes_list = dense_Nodes_list.cpu().numpy()
    
        # for i, node_emb in enumerate(np_Nodes_list):
        #     print(f"node_embs_list[{i}]: {node_emb.shape}")
        # print("node_embs_list", np_Nodes_list.shape)
        
        node_feats = Nodes_list[-1]

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list, Nodes_list)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out, node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
        return out


########## Model for seizure classification/detection ##########
class EvolveGCN_Model_classification(nn.Module):
    def __init__(self, args, num_classes, device=None):
        super(EvolveGCN_Model_classification, self).__init__()

        num_nodes = args.num_nodes
        num_rnn_layers = args.num_rnn_layers
        rnn_units = args.rnn_units
        enc_input_dim = args.input_dim
        max_diffusion_step = args.max_diffusion_step

        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.num_classes = num_classes

        self.encoder = EGCN(args, args.dcgru_activation)

        self.fc = nn.Linear(args.input_dim, num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, input_seq, seq_lengths, supports):
        """
        Args:
            input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
            seq_lengths: actual seq lengths w/o padding, shape (batch,)
            supports: list of supports from laplacian or dual_random_walk filters
        Returns:
            pool_logits: logits from last FC layer (before sigmoid/softmax)
        """
        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]

        # (max_seq_len, batch, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        # init_hidden_state = self.encoder.init_hidden(
        #     batch_size).to(self._device)

        # last hidden state of the encoder is the context
        # (max_seq_len, batch, rnn_units*num_nodes)
        final_hiddens = []
        for i in range(batch_size):
            # バッチのi番目のデータを抽出
            batch_input = input_seq[:, i, :, :]  # 形状: [12, 19, 100]
            batch_supports = supports[i]
            
            # 必要に応じてバッチ次元を追加（例: [1, 12, 19, 100]）
            # batch_input = batch_input.unsqueeze(1)
            
            # print("Batch Input Shape:", batch_input.shape)
            final_hidden = self.encoder(batch_input, batch_supports)
            # print("Final Hidden Shape:", final_hidden.shape)
            
            # final_hiddenをリストに追加
            final_hiddens.append(final_hidden)

        # リストをテンソルに連結
        output = torch.cat([h.unsqueeze(0) for h in final_hiddens], dim=0)
        # print("Output Shape:", output.shape)
        # print("Final Hidden Shape:", final_hidden_cat.shape)
        # (batch_size, max_seq_len, rnn_units*num_nodes)
        # output = torch.transpose(final_hidden, dim0=0, dim1=1)

        # # extract last relevant output
        # last_out = utils.last_relevant_pytorch(
        #     output, seq_lengths, batch_first=True)  # (batch_size, rnn_units*num_nodes)
        # # (batch_size, num_nodes, rnn_units)
        # last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units)
        last_out = output.to(self._device)

        # final FC layer
        logits = self.fc(self.relu(self.dropout(last_out)))
        # print("Logits Shape:", logits.shape)

        # max-pooling over nodes
        pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_classes)
        # print("Pool Logits Shape:", pool_logits.shape)

        return pool_logits, last_out
########## Model for seizure classification/detection ##########


class GRCU(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)

        activation_functions = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }

        self.activation = activation_functions.get(args.activation.lower())
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,A_list,node_embs_list):#,mask_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        # if isinstance(node_embs_list, list):
        #     node_embs_list = torch.stack(node_embs_list, dim=0)  # 新しい次元を追加
        #     print(f"node_embs_list converted to tensor with shape: {node_embs_list.shape}")
        # node_embs_list = node_embs_list.permute(1,0,2,3)
        # print("node: node_embs_list", node_embs_list.shape)
        # print("A_list", A_list.shape)
        # print("")
        for t,Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            #first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.evolve_weights(GCN_weights)#,node_embs,mask_list[t])
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq

class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        
        self.choose_topk = TopK(feats = args.rows,
                                k = args.cols)

    def forward(self,prev_Q):#,prev_Z,mask):
        # z_topk = self.choose_topk(prev_Z,mask)
        z_topk = prev_Q

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

        

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self,feats,k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats,1))
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self,node_embs,mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices,self.k)
            
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1,1))

        #we need to transpose the output
        return out.t()
