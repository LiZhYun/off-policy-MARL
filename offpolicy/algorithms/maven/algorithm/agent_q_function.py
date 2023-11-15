import torch
import torch.nn as nn

from offpolicy.utils.util import to_torch
from offpolicy.algorithms.utils.mlp import MLPBase
from offpolicy.algorithms.utils.act import ACTLayer

class AgentQFunction(nn.Module):
    """
    Individual agent q network (MLP).
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param input_dim: (int) dimension of input to q network
    :param act_dim: (int) dimension of the action space
    :param device: (torch.Device) torch device on which to do computations
    """
    def __init__(self, args, input_dim, act_dim, device):
        super(AgentQFunction, self).__init__()
        self.args = args
        self.act_dim = act_dim
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.mlp = MLPBase(args, input_dim)
        self.q = ACTLayer(act_dim, self.hidden_size, self._use_orthogonal, gain=self._gain)

        self.noise_fc1 = nn.Linear(args.noise_dim + args.num_agents, args.hidden_size)
        self.noise_fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.noise_fc3 = nn.Linear(args.hidden_size, act_dim)

        self.hyper = True
        self.hyper_noise_fc1 = nn.Linear(args.noise_dim + args.num_agents, args.hidden_size * act_dim)

        self.to(device)

    def forward(self, x, noise):
        """
        Compute q values for every action given observations and rnn states.
        :param x: (torch.Tensor) observations from which to compute q values.

        :return q_outs: (torch.Tensor) q values for every action
        """
        # make sure input is a torch tensor
        x = to_torch(x).to(**self.tpdv)
        noise = to_torch(noise).to(**self.tpdv)
        x = self.mlp(x)
        # pass outputs through linear layer
        q_value = self.q(x)

        agent_ids = torch.eye(self.args.num_agents).repeat(noise.shape[0]//self.args.num_agents, 1).to(**self.tpdv)
        # noise_repeated = noise.repeat(1, self.args.num_agents).reshape(agent_ids.shape[0], -1)

        noise_input = torch.cat([noise, agent_ids], dim=-1)

        W = self.hyper_noise_fc1(noise_input).reshape(-1, self.act_dim, self.args.hidden_size)
        wq = torch.bmm(W, x.unsqueeze(2)).squeeze(-1)

        return wq
