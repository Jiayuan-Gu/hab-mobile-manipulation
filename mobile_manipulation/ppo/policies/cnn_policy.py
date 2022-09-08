from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from gym import spaces
from habitat.config.default import Config

# from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    GRUStateEncoder,
    LSTMStateEncoder,
    build_rnn_state_encoder,
)
from torch import nn

from mobile_manipulation.common.registry import mm_registry
from mobile_manipulation.ppo.policy import ActorCritic, CriticHead, Net
from mobile_manipulation.utils.nn_utils import MLP, Flatten


class SimpleCNN(nn.ModuleList):
    def __init__(self, in_channels, input_shape, out_channels) -> None:
        super().__init__()

        self.extend(
            [
                nn.Conv2d(in_channels, 32, 8, stride=4),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(True),
                nn.Conv2d(64, 32, 3, stride=1),
                Flatten(),
            ]
        )

        # Infer the final output resolution
        with torch.no_grad():
            x = torch.zeros(1, in_channels, *input_shape)
            dim = self.forward(x).size(-1)
        self.extend([nn.Linear(dim, out_channels), nn.ReLU(True)])

        self.reset_parameters()

    def forward(self, x):
        for m in self:
            x = m(x)
        return x

    def reset_parameters(self):
        for m in self:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, nn.init.calculate_gain("relu")
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class CRNet(Net):
    """CNN + RNN"""

    visual_encoder: nn.Module
    state_encoder: nn.Module
    rnn_encoder: Union[None, LSTMStateEncoder, GRUStateEncoder]

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: Union[spaces.Discrete, spaces.Box],
        rgb_uuids: List[str],
        depth_uuids: List[str],
        state_uuids: List[str],
        hidden_size: int,
        state_hidden_sizes: List[int],
        rnn_hidden_size: int,
        use_prev_actions=False,
    ) -> None:
        super().__init__()

        self.rgb_uuids = rgb_uuids
        self.depth_uuids = depth_uuids
        self.state_uuids = state_uuids

        self.obs_space = observation_space
        self.action_space = action_space
        self.use_prev_actions = use_prev_actions

        # Visual inputs
        self._n_input_rgb, rgb_shape = self.get_cnn_dim_and_shape(
            observation_space, self.rgb_uuids
        )
        self._n_input_depth, depth_shape = self.get_cnn_dim_and_shape(
            observation_space, self.depth_uuids
        )
        if self._n_input_rgb > 0 and self._n_input_depth > 0:
            assert rgb_shape == depth_shape, (rgb_shape, depth_shape)
        self._n_input_visual = self._n_input_rgb + self._n_input_depth

        # State inputs
        self._n_input_state = self.get_state_dim(
            observation_space, self.state_uuids
        )
        if self.use_prev_actions:
            self._n_input_state += self.get_action_dim(action_space)

        cnn_input_shape = rgb_shape or depth_shape
        self.visual_encoder = SimpleCNN(
            self._n_input_visual, cnn_input_shape, hidden_size
        )

        self.state_encoder = MLP(
            self._n_input_state, state_hidden_sizes
        ).orthogonal_()
        self.output_size = hidden_size + self.state_encoder.output_size

        if rnn_hidden_size > 0:
            self.rnn_encoder = build_rnn_state_encoder(
                self.output_size, rnn_hidden_size
            )
            self.output_size = rnn_hidden_size
            self.rnn_hidden_size = rnn_hidden_size
            self.num_recurrent_layers = self.rnn_encoder.num_recurrent_layers
        else:
            self.rnn_encoder = None
            self.rnn_hidden_size = 0
            self.num_recurrent_layers = 0

    @staticmethod
    def get_cnn_dim_and_shape(obs_spaces: spaces.Dict, uuids: List[str]):
        cnn_dim = 0
        cnn_shape = None
        for uuid in uuids:
            obs_space = obs_spaces[uuid]
            assert isinstance(obs_space, spaces.Box)
            assert len(obs_space.shape) == 3
            cnn_dim += obs_space.shape[2]
            if cnn_shape is None:
                cnn_shape = obs_space.shape[:2]
            else:
                assert cnn_shape == obs_space.shape[:2], obs_space
        return cnn_dim, cnn_shape

    @staticmethod
    def get_state_dim(obs_spaces: spaces.Dict, uuids: List[str]):
        state_dim = 0
        for uuid in uuids:
            obs_space = obs_spaces[uuid]
            assert isinstance(obs_space, spaces.Box)
            assert len(obs_space.shape) == 1
            state_dim += obs_space.shape[0]
        return state_dim

    @staticmethod
    def get_action_dim(action_space: spaces.Space):
        if isinstance(action_space, spaces.Box):
            assert len(action_space.shape) == 1
            return action_space.shape[0]
        elif isinstance(action_space, spaces.Discrete):
            return action_space.n
        else:
            raise NotImplementedError

    def get_cnn_input(self, obs):
        cnn_input = []

        if len(self.rgb_uuids) > 0:
            rgb_obs = [obs[rgb_uuid] for rgb_uuid in self.rgb_uuids]
            rgb_obs = torch.cat(rgb_obs, dim=-1)
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_obs = rgb_obs.permute(0, 3, 1, 2)
            # normalize RGB
            rgb_obs = rgb_obs.float() / 255.0
            cnn_input.append(rgb_obs)

        if len(self.depth_uuids) > 0:
            depth_observations = [
                obs[depth_uuid] for depth_uuid in self.depth_uuids
            ]
            depth_observations = torch.cat(depth_observations, dim=-1)
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        return torch.cat(cnn_input, dim=1)

    def get_state_input(self, obs, prev_actions=None):
        state_input = [obs[uuid] for uuid in self.state_uuids]
        if prev_actions is not None:
            state_input.append(prev_actions)
        return torch.cat(state_input, dim=1)

    def get_prev_actions(self, batch):
        if not self.use_prev_actions:
            return None
        prev_actions = batch["prev_actions"]  # [B, A] or [B, 1]
        masks = batch["masks"]
        if isinstance(self.action_space, spaces.Box):
            pass
        elif isinstance(self.action_space, spaces.Discrete):
            prev_actions = F.one_hot(
                prev_actions.squeeze(-1).long(), self.action_space.n
            ).float()
        else:
            raise NotImplementedError
        return prev_actions * masks

    def forward(self, batch: Dict[str, torch.Tensor]):
        observations = batch["observations"]

        cnn_input = self.get_cnn_input(observations)
        perception_embed = self.visual_encoder(cnn_input)

        state_input = self.get_state_input(
            observations, prev_actions=self.get_prev_actions(batch)
        )
        state_embed = self.state_encoder(state_input)

        features = torch.cat([perception_embed, state_embed], dim=1)
        outputs = dict(features=features)

        if self.rnn_encoder is not None:
            rnn_hidden_states = batch["recurrent_hidden_states"]
            masks = batch["masks"]
            features, rnn_hidden_states = self.rnn_encoder(
                features, rnn_hidden_states, masks
            )
            outputs.update(
                features=features, rnn_hidden_states=rnn_hidden_states
            )

        return outputs


# @baseline_registry.register_policy
@mm_registry.register_policy
class CRPolicy(ActorCritic):
    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        net = CRNet(
            observation_space=observation_space,
            action_space=action_space,
            use_prev_actions=config.get("USE_PREV_ACTIONS", False),
            **config.CRNet
        )

        actor_type = config.get("actor_type", "gaussian")
        if actor_type == "gaussian":
            actor = cls.build_gaussian_actor(
                net.output_size, action_space, **config.GAUSSIAN_ACTOR
            )
        elif actor_type == "categorical":
            actor = cls.build_categorical_actor(
                net.output_size, action_space, **config.CATEGORICAL_ACTOR
            )
        else:
            raise NotImplementedError(actor_type)

        critic = CriticHead(net.output_size, **config.CRITIC)

        return cls(net=net, actor=actor, critic=critic)
