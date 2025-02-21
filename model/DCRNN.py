import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math

import utils
from model.cell import DCGRUCell

VERY_SMALL_NUMBER = 1e-12

class ConcreteGraphLearner(nn.Module):
    """
    Learns a probabilistic adjacency matrix via a Concrete distribution.
    """
    def __init__(self, input_dim, hidden_dim, num_nodes, num_supports=2, temperature=0.1, device='cpu'):
        super(ConcreteGraphLearner, self).__init__()
        self.num_nodes = num_nodes
        self.num_supports = num_supports
        self.temperature = temperature
        self.device = device
        # Create an MLP for each support
        self.node_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_supports)
        ])

    def forward(self, x):
        # x shape: (B, T, N, d)
        # Average over the time dimension => (B, N, d)
        x_agg = x.mean(dim=1)  # (B, N, d)
        B = x_agg.size(0)

        A_list = []
        for head in range(self.num_supports):
            node_embed = self.node_mlps[head](x_agg)  # (B, N, hidden_dim)
            # Compute pi_u,v = sigmoid(Z(u) * Z(v)^T)
            sim = torch.matmul(node_embed, node_embed.transpose(-1, -2))  # (B, N, N)
            pi = torch.sigmoid(sim)  # (B, N, N)

            # Concrete distribution to approximate Bernoulli(pi)
            eps = torch.rand_like(pi)
            eps = torch.clamp(eps, min=VERY_SMALL_NUMBER, max=1 - VERY_SMALL_NUMBER)
            logit = (torch.log(pi + VERY_SMALL_NUMBER) 
                     - torch.log(1. - pi + VERY_SMALL_NUMBER) 
                     + torch.log(eps + VERY_SMALL_NUMBER) 
                     - torch.log(1. - eps + VERY_SMALL_NUMBER))
            A_concrete = torch.sigmoid(logit / self.temperature)  # (B, N, N)

            A_list.append(A_concrete.unsqueeze(1))  # (B, 1, N, N)

        A_batch = torch.cat(A_list, dim=1)  # (B, K, N, N)
        return A_batch


class DCRNNEncoder(nn.Module):
    """
    DCRNN Encoder that uses DCGRUCell layers.
    """
    def __init__(self, input_dim, max_diffusion_step, hid_dim, num_nodes,
                 num_rnn_layers, dcgru_activation=None, filter_type='laplacian', device=None):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers
        self.num_nodes = num_nodes
        self._device = device
        self.num_supports = max_diffusion_step

        cells = []
        # First DCGRUCell layer
        cells.append(
            DCGRUCell(input_dim=input_dim, num_units=hid_dim, max_diffusion_step=max_diffusion_step,
                      num_nodes=num_nodes, nonlinearity=dcgru_activation, filter_type=filter_type)
        )
        # Subsequent DCGRUCell layers
        for _ in range(1, num_rnn_layers):
            cells.append(
                DCGRUCell(input_dim=hid_dim, num_units=hid_dim, max_diffusion_step=max_diffusion_step,
                          num_nodes=num_nodes, nonlinearity=dcgru_activation, filter_type=filter_type)
            )
        self.encoding_cells = nn.ModuleList(cells)

    def forward(self, inputs, initial_hidden_state, supports):
        # inputs shape: (B, T, N, d)
        # Reorder to (T, B, N*d)
        seq_length = inputs.shape[1]
        batch_size = inputs.shape[0]
        inputs = inputs.permute(1, 0, 2, 3).contiguous().view(seq_length, batch_size, -1)

        current_inputs = inputs
        output_hidden = []
        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                input_t = current_inputs[t]
                _, hidden_state = self.encoding_cells[i_layer](
                    supports[:, t, :, :], input_t, hidden_state
                )
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).to(self._device)
        output_hidden = torch.stack(output_hidden, dim=0).to(self._device)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []
        for cell in self.encoding_cells:
            hidden = cell.init_hidden(batch_size)
            init_states.append(hidden)
        return torch.stack(init_states, dim=0)


class DCRNNModel_classification(nn.Module):
    """
    DCRNN-based model for classification with a learnable adjacency matrix.
    """
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

        # ConcreteGraphLearner
        self.graph_learner = ConcreteGraphLearner(
            input_dim=args.input_dim,
            hidden_dim=args.rnn_units,
            num_nodes=args.num_nodes,
            num_supports=2,
            temperature=0.1, 
            device=device
        )

        # Information Bottleneck
        self.IB_size = args.IB_size
        self.mu_layer = nn.Linear(args.rnn_units, self.IB_size)
        self.sigma_layer = nn.Linear(args.rnn_units, self.IB_size)

        self.classifier = nn.Linear(self.IB_size, self.num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def preprocess_adj(self, A):
        # A: (B, K, N, N)
        B, K, N, _ = A.size()
        A_norm = torch.zeros_like(A)
        for k in range(K):
            A_k = A[:, k, :, :]
            d = torch.sum(A_k, dim=-1, keepdim=True)  # degree
            d_inv_sqrt = torch.pow(d + 1e-10, -0.5)
            A_norm_k = A_k * d_inv_sqrt * d_inv_sqrt.transpose(-1, -2)
            A_norm[:, k, :, :] = A_norm_k
        return A_norm

    def forward(self, x, seq_lengths, supports):
        # x: (B, T, N, d)
        A_IB = self.graph_learner(x)  # (B, K, N, N)

        # Normalize adjacency
        new_supports = self.preprocess_adj(A_IB)  # (B, K, N, N)

        # Original supports shape: [B, T, K, N, N]
        # Expand new_supports to [B, T, K, N, N]
        T = supports.shape[1]
        new_supports = new_supports.unsqueeze(1).repeat(1, T, 1, 1, 1)

        # Initialize hidden state
        init_hidden = self.encoder.init_hidden(x.size(0)).to(self._device)

        # Encoder forward pass
        encoder_hidden_state, final_hidden = self.encoder(x, init_hidden, new_supports)

        # final_hidden: (T, B, num_units)
        output = final_hidden.permute(1, 0, 2)  # (B, T, num_units)

        # Use last relevant timestep
        last_out = utils.last_relevant_pytorch(output, seq_lengths, batch_first=True)  # (B, num_units)

        # Reshape for node-level pooling
        last_out = last_out.view(x.size(0), self.num_nodes, -1)  # (B, N, r)
        pooled = torch.max(last_out, dim=1)[0]  # (B, r)

        # IB parameters
        mu = self.mu_layer(pooled)  # (B, IB_size)
        sigma = F.softplus(self.sigma_layer(pooled))  # (B, IB_size)

        # Reparameterization
        eps = torch.randn_like(sigma)
        Z = mu + sigma * eps

        # Classifier
        logits = self.classifier(self.relu(self.dropout(Z)))  # (B, num_classes)

        return logits, Z, mu, sigma


class DCRNNModel_nextTimePred(nn.Module):
    """
    DCRNN-based model for next-step prediction.
    """
    def __init__(self, args, device=None):
        super(DCRNNModel_nextTimePred, self).__init__()

        num_nodes = args.num_nodes
        num_rnn_layers = args.num_rnn_layers
        rnn_units = args.rnn_units
        enc_input_dim = args.input_dim
        dec_input_dim = args.output_dim
        output_dim = args.output_dim
        max_diffusion_step = args.max_diffusion_step

        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.output_dim = output_dim
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = bool(args.use_curriculum_learning)

        self.encoder = DCRNNEncoder(
            input_dim=enc_input_dim,
            max_diffusion_step=max_diffusion_step,
            hid_dim=rnn_units,
            num_nodes=num_nodes,
            num_rnn_layers=num_rnn_layers,
            dcgru_activation=args.dcgru_activation,
            filter_type=args.filter_type,
            device=device
        )

        self.decoder = DCGRUDecoder(
            input_dim=dec_input_dim,
            max_diffusion_step=max_diffusion_step,
            num_nodes=num_nodes,
            hid_dim=rnn_units,
            output_dim=output_dim,
            num_rnn_layers=num_rnn_layers,
            dcgru_activation=args.dcgru_activation,
            filter_type=args.filter_type,
            device=device,
            dropout=args.dropout
        )

    def forward(self, encoder_inputs, decoder_inputs, supports, batches_seen=None):
        """
        Args:
            encoder_inputs: (batch, input_seq_len, num_nodes, input_dim)
            decoder_inputs: (batch, output_seq_len, num_nodes, output_dim)
            supports: adjacency matrices or transition matrices
            batches_seen: for curriculum learning
        Returns:
            outputs: (batch, output_seq_len, num_nodes, output_dim)
        """
        batch_size, output_seq_len, num_nodes, _ = decoder_inputs.shape

        # (seq_len, batch, num_nodes, input_dim)
        encoder_inputs = torch.transpose(encoder_inputs, dim0=0, dim1=1)
        # (seq_len, batch, num_nodes, output_dim)
        decoder_inputs = torch.transpose(decoder_inputs, dim0=0, dim1=1)

        # Initialize encoder hidden state
        init_hidden_state = self.encoder.init_hidden(batch_size).to(self._device)

        # Encoder
        encoder_hidden_state, final_hidden = self.encoder(encoder_inputs, init_hidden_state, supports)

        # Decoder
        if self.training and self.use_curriculum_learning and (batches_seen is not None):
            teacher_forcing_ratio = utils.compute_sampling_threshold(self.cl_decay_steps, batches_seen)
        else:
            teacher_forcing_ratio = None

        outputs = self.decoder(
            decoder_inputs,
            encoder_hidden_state,
            supports,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        # Reshape and transpose back
        outputs = outputs.reshape((output_seq_len, batch_size, num_nodes, -1))
        outputs = torch.transpose(outputs, dim0=0, dim1=1)
        return outputs
