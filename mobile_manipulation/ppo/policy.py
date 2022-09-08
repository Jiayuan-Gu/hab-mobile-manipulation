from typing import Dict, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

# from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import CustomFixedCategorical, CustomNormal
from torch.distributions import Distribution

from mobile_manipulation.utils.nn_utils import MLP


class Net(nn.Module):
    """Base class for backbone to extract features."""

    output_size: int
    rnn_hidden_size = 0
    num_recurrent_layers = 0

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class CategoricalNet(nn.Module):
    def __init__(
        self, num_inputs: int, num_outputs: int, hidden_sizes: Sequence[int]
    ) -> None:
        super().__init__()
        self.mlp = MLP(num_inputs, hidden_sizes).orthogonal_()
        self.linear = nn.Linear(self.mlp.output_size, num_outputs)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


class GaussianNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_sizes: Sequence[int],
        action_activation: str,
        std_transform: str,
        min_std: float,
        max_std: float,
        conditioned_std: bool,
        std_init_bias: float,
        # TODO(jigu): remove deprecated keys in ckpt
        **kwargs
    ) -> None:
        super().__init__()

        assert action_activation in ["", "tanh", "sigmoid"], action_activation
        self.action_activation = action_activation
        assert std_transform in ["log", "softplus"], std_transform
        self.std_transform = std_transform

        self.min_std = min_std
        self.max_std = max_std
        self.conditioned_std = conditioned_std

        self.mlp = MLP(num_inputs, hidden_sizes).orthogonal_()

        self.mu = nn.Linear(self.mlp.output_size, num_outputs)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0)

        if conditioned_std:
            self.std = nn.Linear(self.mlp.output_size, num_outputs)
            nn.init.orthogonal_(self.std.weight, gain=0.01)
            nn.init.constant_(self.std.bias, std_init_bias)
        else:
            self.std = nn.Parameter(
                torch.zeros([num_outputs]), requires_grad=True
            )
            nn.init.constant_(self.std.data, std_init_bias)

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)

        mu = self.mu(x)
        if self.action_activation == "tanh":
            mu = torch.tanh(mu)
        elif self.action_activation == "sigmoid":
            mu = torch.sigmoid(mu)

        std = self.std(x) if self.conditioned_std else self.std
        std = torch.clamp(std, min=self.min_std, max=self.max_std)
        if self.std_transform == "log":
            std = torch.exp(std)
        elif self.std_transform == "softplus":
            std = F.softplus(std)

        return CustomNormal(mu, std)


class CriticHead(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Sequence[int] = ()):
        super().__init__()

        self.mlp = MLP(input_size, hidden_sizes).orthogonal_()

        self.fc = nn.Linear(self.mlp.output_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return self.fc(x)


class ActorCritic(nn.Module):
    r"""Base class for actor-critic policy."""

    def __init__(self, net: Net, actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.net = net
        self.actor = actor
        self.critic = critic

    def act(self, batch: Dict[str, torch.Tensor], deterministic=False):
        net_outputs: Dict[str, torch.Tensor] = self.net(batch)
        features = net_outputs["features"]
        rnn_hidden_states = net_outputs["rnn_hidden_states"]

        distribution: Distribution = self.actor(features)
        value: torch.Tensor = self.critic(features)

        if deterministic:
            if isinstance(distribution, CustomFixedCategorical):
                action = distribution.mode()
            elif isinstance(distribution, CustomNormal):
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return dict(
            action=action,
            action_log_probs=action_log_probs,
            value=value,
            rnn_hidden_states=rnn_hidden_states,
        )

    def get_value(self, batch: Dict[str, torch.Tensor]):
        net_outputs: Dict[str, torch.Tensor] = self.net(batch)
        features = net_outputs["features"]
        return self.critic(features)

    def evaluate_actions(
        self, batch: Dict[str, torch.Tensor], action: torch.Tensor
    ):
        net_outputs: Dict[str, torch.Tensor] = self.net(batch)
        features = net_outputs["features"]
        rnn_hidden_states = net_outputs["rnn_hidden_states"]

        distribution: Distribution = self.actor(features)
        value: torch.Tensor = self.critic(features)

        action_log_probs = distribution.log_probs(action)  # [B, 1]
        dist_entropy = distribution.entropy()  # [B, 1]

        return dict(
            action_log_probs=action_log_probs,
            dist_entropy=dist_entropy,
            value=value,
            rnn_hidden_states=rnn_hidden_states,
        )

    @classmethod
    def build_gaussian_actor(self, num_inputs, action_space, **kwargs):
        assert isinstance(action_space, spaces.Box), action_space
        assert len(action_space.shape) == 1, action_space.shape
        actor = GaussianNet(num_inputs, action_space.shape[0], **kwargs)
        return actor

    @classmethod
    def build_categorical_actor(self, num_inputs, action_space, **kwargs):
        assert isinstance(action_space, spaces.Discrete), action_space
        actor = CategoricalNet(num_inputs, action_space.n, **kwargs)
        return actor
