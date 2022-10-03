#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import deque
from typing import Dict

import numpy as np
import torch
import tqdm
from gym import spaces
from habitat import Config, RLEnv, logger
from habitat.core.environments import get_env_class
from habitat_baselines import BaseTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    batch_obs,
    generate_video,
    get_checkpoint_id,
)
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from mobile_manipulation.common.registry import mm_registry
from mobile_manipulation.common.rollout_storage import RolloutStorage
from mobile_manipulation.ppo.policy import ActorCritic
from mobile_manipulation.utils.common import (
    Timer,
    extract_scalars_from_info,
    get_git_commit_id,
    get_latest_checkpoint,
)
from mobile_manipulation.utils.env_utils import (
    VectorEnv,
    construct_envs,
    make_env_fn,
)
from mobile_manipulation.utils.wrappers import HabitatActionWrapper


@baseline_registry.register_trainer(name="ppo-v0")
class PPOTrainerV0(BaseTrainer):
    r"""Basic PPO Trainer."""

    envs: VectorEnv
    device: torch.device
    actor_critic: ActorCritic
    optimizer: torch.optim.Adam

    obs_space: spaces.Space
    _obs_batching_cache: ObservationBatchingCache
    action_space: spaces.Space

    def __init__(self, config: Config):
        self.config = config

    def is_done(self):
        return self.num_steps_done >= self.config.TOTAL_NUM_STEPS

    def percent_done(self):
        return self.num_steps_done / self.config.TOTAL_NUM_STEPS

    def train(self) -> None:
        ppo_cfg = self.config.RL.PPO

        self._init_train()
        self._init_rollouts()
        self.resume()

        if ppo_cfg.use_linear_lr_decay:
            min_lr = ppo_cfg.get("min_lr", 0.0)
            min_lr_ratio = min_lr / ppo_cfg.lr
            lr_lambda = lambda x: max(
                1 - self.percent_done() * (1.0 - min_lr_ratio), min_lr_ratio
            )
            lr_scheduler = LambdaLR(
                optimizer=self.optimizer, lr_lambda=lr_lambda
            )

        while not self.is_done():
            # Rollout and collect transitions
            self.actor_critic.eval()
            for _ in range(ppo_cfg.num_steps):
                self.step()

            with self.timer.timeit("update_model"):
                self.actor_critic.train()
                losses, metrics = self.update()
                self.rollouts.after_update()

            # Logging
            episode_metrics = self.get_episode_metrics()
            if self.num_updates_done % self.config.LOG_INTERVAL == 0:
                self.log(episode_metrics)

            # Tensorboard
            metrics.update(**episode_metrics)
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            if self.should_summarize():
                self.summarize(losses, metrics)
            if self.should_summarize(10):
                self.summarize2()

            # Checkpoint
            if self.should_checkpoint():
                self.count_checkpoints += 1
                self.prev_ckpt_step = self.num_steps_done
                self.save(ckpt_id=self.count_checkpoints)
            if self.should_checkpoint2():
                self.save(ckpt_id=-1)

            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()

        # Save the last model
        if self.num_steps_done > self.prev_ckpt_step:
            self.count_checkpoints += 1
            self.save(ckpt_id=self.count_checkpoints)

        self.writer.close()
        self.envs.close()

    @torch.no_grad()
    def step(self):
        """Take one rollout step."""
        n_envs = self.envs.num_envs

        with self.timer.timeit("sample_action"):
            step_batch = self.rollouts.buffers[self.rollouts.step_idx]
            # Assume that observations are stored at rollouts in the last step
            output_batch = self.actor_critic.act(step_batch)
            actions = output_batch["action"]
            actions = actions.to(device="cpu", non_blocking=True)

        with self.timer.timeit("step_env"):
            for i_env, action in zip(range(n_envs), actions.unbind(0)):
                self.envs.async_step_at(i_env, {"action": action.numpy()})

        with self.timer.timeit("update_rollout"):
            self.rollouts.insert(
                next_recurrent_hidden_states=output_batch.get(
                    "rnn_hidden_states"
                ),
                actions=output_batch["action"],
                action_log_probs=output_batch["action_log_probs"],
                value_preds=output_batch["value"],
            )

        with self.timer.timeit("step_env"):
            results = self.envs.wait_step()
            obs, rews, dones, infos = map(list, zip(*results))
            self.num_steps_done += n_envs

        # self.envs.render("human", delay=10)

        # -------------------------------------------------------------------------- #
        # Reset and deal with truncated episodes
        # -------------------------------------------------------------------------- #
        next_value = None
        are_truncated = [False for _ in range(n_envs)]
        ignore_truncated = self.config.RL.get("IGNORE_TRUNCATED", False)

        if any(dones):
            # Check which envs are truncated
            for i_env in range(n_envs):
                if dones[i_env]:
                    self.envs.async_reset_at(i_env)
                    is_truncated = infos[i_env].get(
                        "is_episode_truncated", False
                    )
                    are_truncated[i_env] = is_truncated

            if ignore_truncated:
                are_truncated = [False for _ in range(n_envs)]

            # Estimate values of actual next obs
            if any(are_truncated):
                next_batch = batch_obs(
                    obs,
                    device=self.device,
                    cache=self._obs_batching_cache,
                )
                next_step_batch = self.rollouts.buffers[
                    self.rollouts.step_idx + 1
                ]
                next_step_batch["observations"] = next_batch
                # Only the really truncated episodes have valid results
                next_step_batch["masks"] = torch.ones_like(
                    next_step_batch["masks"]
                )
                next_value = self.actor_critic.get_value(next_step_batch)

            for i_env in range(n_envs):
                if dones[i_env]:
                    obs[i_env] = self.envs.wait_reset_at(i_env)
        # -------------------------------------------------------------------------- #

        with self.timer.timeit("batch_obs"):
            batch = batch_obs(
                obs, device=self.device, cache=self._obs_batching_cache
            )
            rewards = torch.tensor(rews, dtype=torch.float).unsqueeze(1)
            done_masks = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)
            not_done_masks = torch.logical_not(done_masks)
            truncated_masks = torch.tensor(
                are_truncated, dtype=torch.bool
            ).unsqueeze(1)

        with self.timer.timeit("update_stats"):
            for i_env in range(n_envs):
                self.episode_rewards[i_env] += rews[i_env]
                if dones[i_env]:
                    episode_info = self._extract_scalars_from_info(
                        infos[i_env]
                    )
                    episode_info["return"] = self.episode_rewards[i_env].item()
                    self.window_episode_stats.append(episode_info)
                    self.episode_rewards[i_env] = 0.0

        with self.timer.timeit("update_rollout"):
            self.rollouts.insert(
                next_observations=batch,
                rewards=rewards,
                next_masks=not_done_masks,
                next_value_preds=next_value,
                truncated_masks=truncated_masks,
            )
            self.rollouts.advance()

    def update(self):
        """PPO update."""
        ppo_cfg = self.config.RL.PPO
        if ppo_cfg.use_linear_clip_decay:
            clip_param = ppo_cfg.clip_param * max(1 - self.percent_done(), 0.0)
        else:
            clip_param = ppo_cfg.clip_param
        ppo_epoch = ppo_cfg.ppo_epoch

        with torch.no_grad():
            step_batch = self.rollouts.buffers[self.rollouts.step_idx]
            next_value = self.actor_critic.get_value(step_batch)

            # NOTE(jigu): next_value will be stored in the buffer.
            # However, it will be overwritten when next action is taken.
            self.rollouts.compute_returns(
                next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
            )
            advantages = self.rollouts.get_advantages(
                ppo_cfg.use_normalized_advantage
            )

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0

        num_updates = 0
        num_clipped_epoch = [0 for _ in range(ppo_epoch)]
        num_samples_epoch = [0 for _ in range(ppo_epoch)]

        for i_epoch in range(ppo_epoch):
            if ppo_cfg.use_recurrent_generator:
                data_generator = self.rollouts.recurrent_generator(
                    advantages, ppo_cfg.num_mini_batch
                )
            else:
                data_generator = self.rollouts.feed_forward_generator(
                    advantages, ppo_cfg.mini_batch_size
                )

            for batch in data_generator:
                outputs = self.actor_critic.evaluate_actions(
                    batch, batch["actions"]
                )
                values = outputs["value"]  # [B, 1]
                action_log_probs = outputs["action_log_probs"]  # [B, 1]
                dist_entropy = outputs["dist_entropy"]  # [B, A]

                ratio = torch.exp(action_log_probs - batch["action_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
                    * batch["advantages"]
                )
                action_loss = -torch.min(surr1, surr2)

                if ppo_cfg.use_clipped_value_loss:
                    value_pred_clipped = batch["value_preds"] + (
                        values - batch["value_preds"]
                    ).clamp(-clip_param, clip_param)
                    value_losses = (values - batch["returns"]).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - batch["returns"]
                    ).pow(2)
                    value_loss = 0.5 * torch.max(
                        value_losses, value_losses_clipped
                    )
                else:
                    value_loss = 0.5 * (batch["returns"] - values).pow(2)

                action_loss = action_loss.mean()
                value_loss = value_loss.mean()
                dist_entropy = dist_entropy.mean()

                self.optimizer.zero_grad()

                # ppo extra metrics
                num_clipped = torch.logical_or(
                    ratio < 1.0 - clip_param,
                    ratio > 1.0 + clip_param,
                ).float()
                num_clipped_epoch[i_epoch] += num_clipped.sum().item()
                num_samples_epoch[i_epoch] += num_clipped.size(0)

                total_loss = (
                    value_loss * ppo_cfg.value_loss_coef
                    + action_loss
                    - dist_entropy * ppo_cfg.entropy_coef
                )
                total_loss.backward()

                if ppo_cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(),
                        ppo_cfg.max_grad_norm,
                    )
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                num_updates += 1

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        loss_dict = dict(
            value_loss=value_loss_epoch,
            action_loss=action_loss_epoch,
            dist_entropy=dist_entropy_epoch,
        )

        metric_dict = dict()
        for i_epoch in range(ppo_epoch):
            clip_ratio = num_clipped_epoch[i_epoch] / max(
                num_samples_epoch[i_epoch], 1
            )
            metric_dict[f"clip_ratio_{i_epoch}"] = clip_ratio

        self.num_updates_done += 1
        return loss_dict, metric_dict

    def get_episode_metrics(self):
        if len(self.window_episode_stats) == 0:
            return {}
        # Assume all episodes have the same keys. True for Habitat.
        return {
            k: np.mean([ep_info[k] for ep_info in self.window_episode_stats])
            for k in self.window_episode_stats[0].keys()
        }

    @classmethod
    def _extract_scalars_from_info(cls, info: Dict):
        return extract_scalars_from_info(
            info, blacklist=["terminal_observation"]
        )

    def log(self, metrics: Dict[str, float]):
        wall_time = (time.time() - self.t_start) + self.prev_time
        logger.info(
            "update: {}\tframes: {}\tfps: {:.3f}\tpercent: {:.2f}%".format(
                self.num_updates_done,
                self.num_steps_done,
                self.num_steps_done / wall_time,
                self.percent_done() * 100,
            )
        )
        logger.info(
            "\t".join(
                "{}: {:.3f}s".format(k, v)
                for k, v in self.timer.elapsed_times.items()
            )
        )
        logger.info(
            "  ".join("{}: {:.3f}".format(k, v) for k, v in metrics.items()),
        )

    def summarize(self, losses: Dict[str, float], metrics: Dict[str, float]):
        """Summarize scalars in tensorboard."""
        for k, v in losses.items():
            self.writer.add_scalar(f"losses/{k}", v, self.num_steps_done)
        for k, v in metrics.items():
            self.writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)

    def summarize2(self):
        """Summarize histogram and video in tensorboard."""
        self.writer.add_histogram(
            "value_preds",
            self.rollouts.buffers["value_preds"],
            global_step=self.num_steps_done,
        )
        self.writer.add_histogram(
            "discounted_returns",
            self.rollouts.buffers["returns"],
            global_step=self.num_steps_done,
        )

        video_keys = self.config.get("TB_VIDEO_KEYS", [])
        for key in video_keys:
            video_tensor = self.rollouts.buffers["observations"][key]
            self.writer.add_video(
                key,
                video_tensor.permute(1, 0, 4, 2, 3),
                global_step=self.num_steps_done,
                fps=10,
            )

    def should_summarize(self, mult=1) -> bool:
        if self.config.SUMMARIZE_INTERVAL == -1:
            interval = self.config.LOG_INTERVAL
        else:
            interval = self.config.SUMMARIZE_INTERVAL
        return self.num_updates_done % (interval * mult) == 0

    def should_checkpoint(self) -> bool:
        if self.config.NUM_CHECKPOINTS == -1:
            ckpt_freq = self.config.CHECKPOINT_INTERVAL
        else:
            ckpt_freq = (
                self.config.TOTAL_NUM_STEPS // self.config.NUM_CHECKPOINTS
            )
        return self.num_steps_done >= (self.count_checkpoints + 1) * ckpt_freq

    def should_checkpoint2(self) -> bool:
        """Check whether to save (overwrite) the latest checkpoint."""
        if (
            self.config.NUM_CHECKPOINTS == -1
            or self.config.CHECKPOINT_INTERVAL == -1
        ):
            return False
        return self.num_updates_done % self.config.CHECKPOINT_INTERVAL == 0

    def save_checkpoint(self, ckpt_path):
        wall_time = (time.time() - self.t_start) + self.prev_time
        checkpoint = dict(
            config=self.config,
            state_dict=self.actor_critic.state_dict(),
            optim_state=self.optimizer.state_dict(),
            step=self.num_steps_done,
            wall_time=wall_time,
            num_updates_done=self.num_updates_done,
            count_checkpoints=self.count_checkpoints,
        )
        torch.save(checkpoint, ckpt_path)

    def save(self, ckpt_id):
        if not self.config.CHECKPOINT_FOLDER:
            return
        ckpt_path = os.path.join(
            self.config.CHECKPOINT_FOLDER, f"ckpt.{ckpt_id}.pth"
        )
        self.save_checkpoint(ckpt_path)
        logger.info(
            f"Saved checkpoint to {ckpt_path} at {self.num_steps_done}th step"
        )

    def resume(self):
        if not self.config.CHECKPOINT_FOLDER:
            return
        ckpt_path = get_latest_checkpoint(self.config.CHECKPOINT_FOLDER, False)
        if ckpt_path is None:
            return
        assert os.path.isfile(ckpt_path), ckpt_path
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        logger.info(f"Resume from {ckpt_path}")

        self.actor_critic.load_state_dict(ckpt_dict["state_dict"])
        self.optimizer.load_state_dict(ckpt_dict["optim_state"])

        self.num_steps_done = ckpt_dict["step"]
        self.num_updates_done = ckpt_dict["num_updates_done"]
        self.prev_time = ckpt_dict["wall_time"]
        self.count_checkpoints = ckpt_dict["count_checkpoints"]
        self.prev_ckpt_step = self.num_steps_done

    def _init_envs(self, config: Config, auto_reset_done=False):
        r"""Initialize vectorized environments."""
        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            split_dataset=config.get("SPLIT_DATASET", False),
            workers_ignore_signals=False,
            auto_reset_done=auto_reset_done,
            wrappers=[HabitatActionWrapper],
        )

    def _init_observation_space(self, config: Config):
        if isinstance(self.envs, VectorEnv):
            obs_space = self.envs.observation_spaces[0]
        else:
            env: RLEnv = self.envs[0]
            obs_space = env.observation_space
        self.obs_space = obs_space

    def _init_action_space(self, config: Config):
        if isinstance(self.envs, VectorEnv):
            self.action_space = self.envs.action_spaces[0]
        else:
            env: RLEnv = self.envs[0]
            self.action_space = env.action_space

    def _setup_actor_critic(self, config: Config) -> None:
        r"""Set up actor critic for PPO."""
        policy_cfg = config.RL.POLICY
        # policy = baseline_registry.get_policy(policy_cfg.name)
        policy = mm_registry.get_policy(policy_cfg.name)
        self.actor_critic: ActorCritic = policy.from_config(
            policy_cfg, self.obs_space, self.action_space
        )
        self.actor_critic.to(self.device)

        ppo_cfg = config.RL.PPO
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=ppo_cfg.lr, eps=ppo_cfg.eps
        )

        ckpt_path = self.config.RL.POLICY.get("pretrained_weights", None)
        if ckpt_path:
            assert os.path.isfile(ckpt_path), ckpt_path
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")
            logger.info("Load checkpoint from {}".format(ckpt_path))
            self.actor_critic.load_state_dict(ckpt_dict["state_dict"])

    def _setup_rollouts(self, config: Config):
        ppo_cfg = config.RL.PPO
        self.rollouts = RolloutStorage(
            ppo_cfg.num_steps,  # number of steps for each env
            self.envs.num_envs,
            observation_space=self.obs_space,
            action_space=self.action_space,
            recurrent_hidden_state_size=self.actor_critic.net.rnn_hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
        )
        self.rollouts.to(self.device)

    def _init_train(self):
        if self.config.LOG_FILE:
            log_dir = os.path.dirname(self.config.LOG_FILE)
            os.makedirs(log_dir, exist_ok=True)
            logger.add_filehandler(self.config.LOG_FILE)

        if self.config.VERBOSE:
            logger.info(f"config:\n {self.config}")
            logger.info("commit id: {}".format(get_git_commit_id()))

        if self.config.CHECKPOINT_FOLDER:
            os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)

        # ---------------------------------------------------------------------------- #
        # Initialization
        # ---------------------------------------------------------------------------- #
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # ---------------------------------------------------------------------------- #
        # NOTE(jigu): workaround from erik, to avoid high gpu memory fragmentation
        env = make_env_fn(
            self.config,
            get_env_class(self.config.ENV_NAME),
            wrappers=[HabitatActionWrapper],
        )
        self.envs = [env]
        self._init_observation_space(self.config)
        self._init_action_space(self.config)
        self._setup_actor_critic(self.config)
        env.close()
        # ---------------------------------------------------------------------------- #

        self._init_envs(self.config)
        self._init_observation_space(self.config)
        self._init_action_space(self.config)
        self._setup_rollouts(self.config)

        if self.config.VERBOSE:
            logger.info(f"actor_critic: {self.actor_critic}")
        logger.info(
            "#parameters: {}".format(
                sum(param.numel() for param in self.actor_critic.parameters())
            )
        )
        logger.info("obs space: {}".format(self.obs_space))
        logger.info("action space: {}".format(self.action_space))

        # ---------------------------------------------------------------------------- #
        # Setup statistic
        # ---------------------------------------------------------------------------- #
        # Current episode rewards (return)
        self.episode_rewards = torch.zeros(self.envs.num_envs, 1)
        # Recent episode stats (each stat is a dict)
        self.window_episode_stats = deque(
            maxlen=self.config.RL.PPO.reward_window_size
        )

        self.t_start = time.time()  # record overall time
        self.timer = Timer()  # record fine-grained scopes
        self.writer = TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=30
        )

        # resumable stats
        self.num_steps_done = 0
        self.num_updates_done = 0
        self.prev_time = 0.0
        self.count_checkpoints = 0
        self.prev_ckpt_step = 0

    def _init_rollouts(self):
        self._obs_batching_cache = ObservationBatchingCache()
        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        self.rollouts.buffers["observations"][0] = batch

    def eval(self):
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        if self.config.LOG_FILE:
            log_dir = os.path.dirname(self.config.LOG_FILE)
            os.makedirs(log_dir, exist_ok=True)
            logger.add_filehandler(self.config.LOG_FILE)

        writer = TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=30)

        if self.config.EVAL.CKPT_PATH:
            ckpt_path = self.config.EVAL.CKPT_PATH
        else:
            ckpt_path = get_latest_checkpoint(
                self.config.CHECKPOINT_FOLDER, True
            )
        assert os.path.isfile(ckpt_path), ckpt_path
        ckpt_id = get_checkpoint_id(ckpt_path)
        if ckpt_id is None:
            ckpt_id = -1

        if self.config.EVAL.BATCH_ENVS:
            self._batch_eval_checkpoint(ckpt_path, writer, ckpt_id)
        else:
            self._eval_checkpoint(ckpt_path, writer, ckpt_id)
        writer.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = -1,
    ) -> None:
        # Map location CPU is almost always better than mapping to a CUDA device.
        logger.info(f"Loaded {checkpoint_path}")
        ckpt_dict = torch.load(checkpoint_path, map_location="cpu")

        config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if config.VERBOSE:
            logger.info(config)

        env = make_env_fn(
            config,
            get_env_class(config.ENV_NAME),
            wrappers=[HabitatActionWrapper],
        )
        self.envs = [env]
        self._init_observation_space(config)
        self._init_action_space(config)
        self._setup_actor_critic(config)
        self.actor_critic.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic.eval()

        if config.EVAL.NUM_EPISODES == -1:
            num_eval_episodes = env.number_of_episodes
        else:
            num_eval_episodes = config.EVAL.NUM_EPISODES

        current_episode_reward = 0.0
        all_episode_stats = []
        rgb_frames = []
        failure_episodes = []

        # Initialize policy inputs
        obs = env.reset()
        self._obs_batching_cache = ObservationBatchingCache()
        batch = batch_obs(
            [obs], device=self.device, cache=self._obs_batching_cache
        )
        buffer = dict(
            recurrent_hidden_states=torch.zeros(
                1,
                self.actor_critic.net.num_recurrent_layers,
                self.actor_critic.net.rnn_hidden_size,
                device=self.device,
            ),
            prev_actions=torch.zeros(
                1,
                *self.action_space.shape,
                device=self.device,
                dtype=torch.float,
            ),
            masks=torch.zeros(
                1,
                1,
                device=self.device,
                dtype=torch.bool,
            ),
        )

        metrics = {}

        pbar = tqdm.tqdm(total=num_eval_episodes)
        while len(all_episode_stats) < num_eval_episodes:
            if len(config.VIDEO_OPTION) > 0:
                rgb_frames.append(env.render("human", info=metrics))

            with torch.no_grad():
                step_batch = dict(observations=batch, **buffer)
                outputs_batch = self.actor_critic.act(
                    step_batch, deterministic=config.EVAL.DETERMINISTIC_ACTION
                )
                actions = outputs_batch["action"]

            # step_action = {"action": actions[0].cpu().numpy()}
            step_action = actions[0].cpu().numpy()
            obs, reward, done, info = env.step(step_action)
            current_episode_reward += reward
            metrics = self._extract_scalars_from_info(info)

            if done:
                episode_stats = metrics.copy()
                episode_stats["return"] = current_episode_reward
                all_episode_stats.append(episode_stats)
                pbar.update()

                success_measure = self.config.RL.SUCCESS_MEASURE
                if success_measure in info:
                    episode_success = info[success_measure]
                    if not episode_success:
                        failure_episodes.append(env.current_episode.episode_id)
                else:
                    episode_success = -1

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames,
                        episode_id=env.current_episode.episode_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={"success": episode_success},
                        tb_writer=writer,
                        fps=30,
                    )

                obs = env.reset()
                metrics = {}
                current_episode_reward = 0
                rgb_frames = []

            # Update policy inputs
            batch = batch_obs(
                [obs], device=self.device, cache=self._obs_batching_cache
            )
            not_done_masks = torch.tensor(
                [[not done]], dtype=torch.bool, device=self.device
            )
            buffer.update(
                recurrent_hidden_states=outputs_batch["rnn_hidden_states"],
                prev_actions=outputs_batch["action"],
                masks=not_done_masks,
            )

        # Logging metrics
        aggregated_stats = {
            k: np.mean([ep_info[k] for ep_info in all_episode_stats])
            for k in all_episode_stats[0].keys()
        }
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        failure_episodes = sorted(failure_episodes)
        failure_episodes_str = ",".join(map(str, failure_episodes))
        logger.info("Failure episodes:\n{}".format(failure_episodes_str))

        # Summarize in tensorboard
        step_id = ckpt_dict.get("step", checkpoint_index)
        for k, v in aggregated_stats.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        env.close()

    def _batch_eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = -1,
    ) -> None:
        """Evaluate the checkpoint with a batch of envs.
        Videos are not supported for simplicity.
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        logger.info(f"Loaded {checkpoint_path}")
        ckpt_dict = torch.load(checkpoint_path, map_location="cpu")

        config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if config.VERBOSE:
            logger.info(config)

        self._init_envs(config, auto_reset_done=True)
        self._init_observation_space(config)
        self._init_action_space(config)
        self._setup_actor_critic(config)
        self.actor_critic.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic.eval()

        if config.EVAL.NUM_EPISODES == -1:
            num_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            num_eval_episodes = config.EVAL.NUM_EPISODES

        num_envs = self.envs.num_envs
        current_episode_rewards = [0 for _ in range(num_envs)]
        all_episode_stats = dict()
        failure_episodes = []

        # Initialize policy inputs
        obs = self.envs.reset()
        self._obs_batching_cache = ObservationBatchingCache()
        batch = batch_obs(
            obs, device=self.device, cache=self._obs_batching_cache
        )
        buffer = dict(
            recurrent_hidden_states=torch.zeros(
                num_envs,
                self.actor_critic.net.num_recurrent_layers,
                self.actor_critic.net.rnn_hidden_size,
                device=self.device,
            ),
            prev_actions=torch.zeros(
                num_envs,
                *self.action_space.shape,
                device=self.device,
                dtype=torch.float,
            ),
            masks=torch.zeros(
                num_envs,
                1,
                device=self.device,
                dtype=torch.bool,
            ),
        )

        pbar = tqdm.tqdm(total=num_eval_episodes)
        while len(all_episode_stats) < num_eval_episodes:
            current_episodes = self.envs.current_episodes()
            with torch.no_grad():
                step_batch = dict(observations=batch, **buffer)
                outputs_batch = self.actor_critic.act(
                    step_batch, deterministic=config.EVAL.DETERMINISTIC_ACTION
                )
                actions = outputs_batch["action"]
                actions = actions.to(device="cpu", non_blocking=True)

            step_action = [{"action": a.numpy()} for a in actions]
            results = self.envs.step(step_action)
            obs, rewards, dones, infos = zip(*results)

            for i_env in range(num_envs):
                current_episode_rewards[i_env] += rewards[i_env]

                if dones[i_env]:
                    episode_id = current_episodes[i_env].episode_id
                    # print("Episode {} done".format(episode_id))

                    # Ignore if the episode has already been evaluated
                    if episode_id not in all_episode_stats:
                        metrics = self._extract_scalars_from_info(infos[i_env])
                        episode_stats = metrics.copy()
                        episode_stats["return"] = current_episode_rewards[
                            i_env
                        ]
                        all_episode_stats[episode_id] = episode_stats

                        success_measure = self.config.RL.SUCCESS_MEASURE
                        if success_measure in infos[i_env]:
                            episode_success = infos[i_env][success_measure]
                            if not episode_success:
                                failure_episodes.append(episode_id)
                        else:
                            episode_success = -1

                        pbar.update()

                    # Reset episode stats
                    current_episode_rewards[i_env] = 0

            # Update policy inputs
            batch = batch_obs(
                obs, device=self.device, cache=self._obs_batching_cache
            )
            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.device,
            )
            buffer.update(
                recurrent_hidden_states=outputs_batch["rnn_hidden_states"],
                prev_actions=outputs_batch["action"],
                masks=not_done_masks,
            )

        # Logging metrics
        episode_ids = list(all_episode_stats.keys())
        stat_keys = list(all_episode_stats[episode_ids[0]].keys())
        aggregated_stats = {
            k: np.mean([ep_info[k] for ep_info in all_episode_stats.values()])
            for k in stat_keys
        }
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        # Logging failure episodes
        failure_episodes = sorted(failure_episodes)
        failure_episodes_str = ",".join(map(str, failure_episodes))
        logger.info("Failure episodes:\n{}".format(failure_episodes_str))

        # Summarize in tensorboard
        step_id = ckpt_dict.get("step", checkpoint_index)
        for k, v in aggregated_stats.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
