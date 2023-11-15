import torch
import torch.nn as nn
from offpolicy.utils.util import init, to_torch
from offpolicy.algorithms.utils.mlp import MLPBase
from offpolicy.algorithms.utils.act import ACTLayer

class MACPF_Actor(nn.Module):
    def __init__(self, args, obs_dim, act_dim, device):
        """
        Actor network class for MADDPG/MATD3. Outputs actions given observations.
        :param args: (argparse.Namespace) arguments containing relevant model information.
        :param obs_dim: (int) dimension of the observation vector.
        :param act_dim: (int) dimension of the action vector.
        :param device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(MACPF_Actor, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        # map observation input into input for rnn
        self.mlp = MLPBase(args, obs_dim)

        # get action from rnn hidden state
        self.act = ACTLayer(act_dim, self.hidden_size, self._use_orthogonal, self._gain)

        self.dep_fc1 = nn.Linear(act_dim * args.num_agents, self.hidden_size)
        self.dep_fc2 = nn.Linear(self.hidden_size + self.hidden_size, act_dim)

        self.to(device)

    def forward(self, x, parents_actions=None, execution_mask=None, dep_mode=False):
        """
        Compute actions using the needed information.
        :param x: (np.ndarray) Observations with which to compute actions.
        """
        x = to_torch(x).to(**self.tpdv)
        if parents_actions is not None:
            parents_actions = to_torch(parents_actions).to(**self.tpdv)
        if execution_mask is not None:
            execution_mask = to_torch(execution_mask).to(**self.tpdv)
        x = self.mlp(x)
        # pass outputs through linear layer
        action = self.act(x)

        if parents_actions is not None:
            dep_in = self.dep_fc1((parents_actions * execution_mask.unsqueeze(-1)).view(*parents_actions.shape[:-2], -1))
            if dep_mode:
                dep_final = torch.cat([x, dep_in], dim=-1)
                dep_action = self.dep_fc2(dep_final)
                action = action.detach() + dep_action

        return action


class MACPF_Critic(nn.Module):
    """
    Critic network class for MADDPG/MATD3. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param central_obs_dim: (int) dimension of the centralized observation vector.
    :param central_act_dim: (int) dimension of the centralized action vector.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    :param num_q_outs: (int) number of q values to output (1 for MADDPG, 2 for MATD3).
    """
    def __init__(self, args, obs_dim, act_dim, device, num_q_outs=1):
        super(MACPF_Critic, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        input_dim = obs_dim+act_dim

        self.mlp = MLPBase(args, input_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        self.q_outs = init_(nn.Linear(self.hidden_size, 1))

        self.dep_fc1 = nn.Linear(act_dim * args.num_agents, self.hidden_size)
        self.dep_fc2 = nn.Linear(self.hidden_size + self.hidden_size, 1)
        
        self.to(device)

    def forward(self, obs, act, parents_actions=None, execution_mask=None, dep_mode=False):
        """
        Compute Q-values using the needed information.
        :param central_obs: (np.ndarray) Centralized observations with which to compute Q-values.
        :param central_act: (np.ndarray) Centralized actions with which to compute Q-values.

        :return q_values: (list) Q-values outputted by each Q-network.
        """
        if parents_actions is not None:
            parents_actions = to_torch(parents_actions).to(**self.tpdv)
        if execution_mask is not None:
            execution_mask = to_torch(execution_mask).to(**self.tpdv)
        obs = to_torch(obs).to(**self.tpdv)
        act = to_torch(act).to(**self.tpdv)

        x = torch.cat([obs, act], dim=1)

        x = self.mlp(x)
        q_value = self.q_outs(x)

        if parents_actions is not None:
            dep_in = self.dep_fc1((parents_actions * execution_mask.unsqueeze(-1)).view(*parents_actions.shape[:-2], -1))
            if dep_mode:
                dep_final = torch.cat([x, dep_in], dim=-1)
                dep_value = self.dep_fc2(dep_final)
                # for q_value in q_value:
                q_value = q_value.detach() + dep_value

        return q_value
