import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from offpolicy.utils.util import init, check
from offpolicy.algorithms.utils.rnn import RNNBase
from offpolicy.algorithms.utils.act import ACTLayer

class R_Critic(nn.Module):
    def __init__(self, args, central_obs_dim, central_act_dim, device):
        super(R_Critic, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        input_dim = central_obs_dim + central_act_dim

        self.rnn = RNNBase(args, input_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.q1_out = init_(nn.Linear(self.hidden_size, 1))
        self.q2_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(self.device)

    def forward(self, central_obs, central_act, rnn_states):
        # ensure inputs are torch tensors
        central_obs = check(central_obs).to(**self.tpdv)
        central_act = check(central_act).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)

        no_sequence = False
        if len(central_obs.shape) == 2 and len(central_act.shape) == 2:
            # no sequence, so add a time dimension of len 0
            no_sequence = True
            central_obs, central_act = central_obs[None], central_act[None]

        if len(rnn_states.shape) == 2:
            # also add a first dimension to the rnn hidden states
            rnn_states = rnn_states[None]

        inp = torch.cat([central_obs, central_act], dim=2)

        rnn_outs, h_final = self.rnn(inp, rnn_states)

        q1_values = self.q1_out(rnn_outs)
        q2_values = self.q2_out(rnn_outs)

        if no_sequence:
            # remove the time dimension
            q1_values = q1_values[0, :, :]
            q2_values = q2_values[0, :, :]

        return q1_values, q2_values, h_final


class R_Actor(nn.Module):
    def __init__(self, args, obs_dim, act_dim, device, take_prev_action=False):
        super(R_DiscreteActor, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        self.hidden_size = args.hidden_size
        self.device = device
        self.take_prev_act = take_prev_action
        self.tpdv = dict(dtype=torch.float32, device=device)

        input_dim = (obs_dim + act_dim) if take_prev_action else obs_dim

        # map observation input into input for rnn
        self.rnn = RNNBase(args, input_dim)

        # get action from rnn hidden state
        self.act = ACTLayer(act_dim, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, prev_acts, rnn_states):
        # make sure input is a torch tensor
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        if prev_acts is not None:
            prev_acts = check(prev_acts).to(**self.tpdv)       

        no_sequence = False
        if len(obs.shape) == 2:
            # this means we're just getting one output (no sequence)
            no_sequence = True
            obs = obs[None]
            if self.take_prev_act:
                prev_acts = prev_acts[None]
            # obs is now of shape (seq_len, batch_size, obs_dim)
        if len(rnn_states.shape) == 2:
            # hiddens should be of shape (1, batch_size, dim)
            rnn_states = rnn_states[None]

        inp = torch.cat((obs, prev_acts), dim=-1) if self.take_prev_act else obs

        rnn_outs, h_final = self.rnn(inp, rnn_states)
        # pass outputs through linear layer
        act_outs = self.act(rnn_outs, no_sequence)

        return act_outs, h_final