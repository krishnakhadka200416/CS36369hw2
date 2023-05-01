import copy
import pickle
import random
import gymnasium as gym
import torch
from collections import deque, namedtuple
from gymnasium.utils.save_video import save_video
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
from utils import *
import torch.nn.functional as F


import torch.nn.utils.rnn as rnn_utils
torch.autograd.set_detect_anomaly(True)

# Class for training an RL agent with Actor-Critic
class ACTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.agent = ACAgent(env=self.env, params=self.params)
        self.actor_net = ActorNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.critic_net = CriticNet(input_size=self.env.observation_space.shape[0], output_size=1, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.actor_optimizer = Adam(params=self.actor_net.parameters(), lr=self.params['actor_lr'])
        self.critic_optimizer = Adam(params=self.critic_net.parameters(), lr=self.params['critic_lr'])
        self.trajectory = None

    def run_training_loop(self):
        list_ro_reward = []
        for ro_idx in range(self.params['n_rollout']):
            self.trajectory = self.agent.collect_trajectory(policy=self.actor_net)
            self.update_critic_net()
            self.estimate_advantage()
            self.update_actor_net()

            # Calculate average reward for this rollout
            rewards_per_trajectory = [sum(self.trajectory['reward'][t]) for t in range(self.params['n_trajectory_per_rollout'])]
            avg_ro_reward = sum(rewards_per_trajectory) / self.params['n_trajectory_per_rollout']

            print(f'End of rollout {ro_idx}: Average trajectory reward is {avg_ro_reward: 0.2f}')

            # Append average rollout reward into a list
            list_ro_reward.append(avg_ro_reward)

        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ro_reward, f)

        # Save a video of the trained agent playing
        self.generate_video()

        # Close environment
        self.env.close()



    def update_critic_net(self):
        for critic_iter_idx in range(self.params['n_critic_iter']):
            self.update_target_value()
            for critic_epoch_idx in range(self.params['n_critic_epoch']):
                critic_loss = self.estimate_critic_loss_function()
                critic_loss.backward(retain_graph=True)
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()


    def update_target_value(self, gamma=0.99):
        observations = self.trajectory['obs']
        path_rewards = self.trajectory['reward']
        path_tgt_values = []
        for idx, obs in enumerate(observations):
            rewards_list = path_rewards[idx]
            rewards_arr = np.array(rewards_list, dtype=np.float32).reshape(-1, 1)
            next_states = np.vstack((obs[1:], np.zeros((1, obs.shape[1]))))
            next_state_vals_arr = self.critic_net(torch.tensor(next_states, dtype=torch.float32)).detach().numpy()
            tgt_vals_arr = rewards_arr + gamma * next_state_vals_arr
            path_tgt_values.append(torch.tensor(tgt_vals_arr, dtype=torch.float32))

        self.trajectory['target_value'] = path_tgt_values

        

    def estimate_advantage(self, gamma=0.99):
        # TODO: Estimate advantage
        # HINT: Use definition of advantage-estimate from equation 6 of teh assignment PDF
        observations = self.trajectory['obs']
        self.update_target_value()
        path_state_vals = [self.critic_net(obs) for obs in observations]
        self.trajectory['state_value'] = path_state_vals
        state_vals = self.trajectory['state_value']
        tgt_vals = self.trajectory['target_value']
        advantage_vals = []

        for state_vals_idx, tgt_vals_idx in zip(state_vals, tgt_vals):
            adv_vals_idx = tgt_vals_idx - state_vals_idx
            advantage_vals.append(adv_vals_idx.detach())
        self.trajectory['advantage'] = advantage_vals


    def update_actor_net(self):
        actor_loss = self.estimate_actor_loss_function()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad() 


    def estimate_critic_loss_function(self):
        obs = self.trajectory['obs']
        traj_states = []
        for t_idx in range(len(obs)):
            traj_states.append(self.critic_net(obs[t_idx]))
        self.trajectory['state_value'] = traj_states
        target_values = torch.cat(self.trajectory['target_value'])
        state_values = torch.cat(self.trajectory['state_value'])
        critic_loss = torch.mean((target_values - state_values) ** 2)
        return critic_loss



    def estimate_actor_loss_function(self):
        actor_loss = list()
        log_probs_list = self.trajectory['log_prob']
        for t_idx in range(self.params['n_trajectory_per_rollout']):
            log_probs_list_idx = log_probs_list[t_idx]
            discounted_advantage = apply_discount([t.item() for t in self.trajectory['advantage'][t_idx]])
            # TODO: Compute actor loss function
            loss_value = sum(log_prob * adv_val for log_prob, adv_val in zip(log_probs_list_idx, discounted_advantage))
            actor_loss.append(-loss_value)

        actor_loss = torch.stack(actor_loss).mean()
        return actor_loss
    


    def generate_video(self, max_frame=1000):
        self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
        obs, _ = self.env.reset()
        for _ in range(max_frame):
            action_idx, log_prob = self.actor_net(torch.tensor(obs, dtype=torch.float32, device=get_device()))
            obs, reward, terminated, truncated, info = self.env.step(self.agent.action_space[action_idx.item()])
            if terminated or truncated:
                break
        save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)


# CLass for actor-net
class ActorNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(ActorNet, self).__init__()
        # TODO: Define the actor net
        # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
        self.ff_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_size),
            nn.Softmax(dim = -1)
        )

    def forward(self, obs):
        # TODO: Forward pass of actor net
        # HINT: (use Categorical from torch.distributions to draw samples and log-prob from model output)
        logits = self.ff_net(obs)
        dist = Categorical(logits=logits)
        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)
        return action_index, log_prob


# CLass for CriticNet
class CriticNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(CriticNet, self).__init__()
        # TODO: Define the critic net
        # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
        self.ff_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, obs):
        # TODO: Forward pass of critic net
        # HINT: (get state value from the network using the current observation)
        state_value = self.ff_net(obs)
        return state_value


# Class for agent
class ACAgent:
    def __init__(self, env, params=None):
        self.env = env
        self.params = params
        self.action_space = [action for action in range(self.env.action_space.n)]



    def collect_trajectory(self, policy):
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        rollout_buffer = list()
        for _ in range(self.params['n_trajectory_per_rollout']):
            trajectory_buffer = {'obs': list(), 'log_prob': list(), 'reward': list()}
            while True:
                obs = torch.tensor(obs, dtype=torch.float32, device=get_device())
                # Save observation
                trajectory_buffer['obs'].append(obs)
                action_idx, log_prob = policy(obs)
                obs, reward, terminated, truncated, info = self.env.step(self.action_space[action_idx.item()])
                # Save log-prob and reward into the buffer
                trajectory_buffer['log_prob'].append(log_prob)
                trajectory_buffer['reward'].append(reward)
                # Check for termination criteria
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    rollout_buffer.append(trajectory_buffer)
                    break
        rollout_buffer = self.serialize_trajectory(rollout_buffer)
        return rollout_buffer

    # Converts a list-of-dictionary into dictionary-of-list
    @staticmethod
    def serialize_trajectory(rollout_buffer):
        serialized_buffer = {'obs': list(), 'log_prob': list(), 'reward': list()}
        for trajectory_buffer in rollout_buffer:
            serialized_buffer['obs'].append(torch.stack(trajectory_buffer['obs']))
            serialized_buffer['log_prob'].append(torch.stack(trajectory_buffer['log_prob']))
            serialized_buffer['reward'].append(trajectory_buffer['reward'])
        return serialized_buffer


class DQNTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.q_net = QNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.target_net = QNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = self.params['init_epsilon']
        self.optimizer = Adam(params=self.q_net.parameters(), lr=self.params['lr'])
        self.replay_memory = ReplayMemory(capacity=self.params['rm_cap'])

    def run_training_loop(self):
        list_ep_reward = list()
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        for idx_episode in range(self.params['n_episode']):
            ep_len = 0
            while True:
                ep_len += 1
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    self.epsilon = max(self.epsilon*self.params['epsilon_decay'], self.params['min_epsilon'])
                    next_obs = None
                    self.replay_memory.push(obs, action, reward, next_obs, not (terminated or truncated))
                    list_ep_reward.append(ep_len)
                    print(f'End of episode {idx_episode} with epsilon = {self.epsilon: 0.2f} and reward = {ep_len}, memory = {len(self.replay_memory.buffer)}')
                    obs, _ = self.env.reset()
                    break
                self.replay_memory.push(obs, action, reward, next_obs, not (terminated or truncated))
                obs = copy.deepcopy(next_obs)
                self.update_q_net()
                self.update_target_net()
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ep_reward, f)
        # Save a video of the trained agent playing
        self.generate_video()
        # Close environment
        self.env.close()

    def get_action(self, obs):
        # TODO: Implement the epsilon-greedy behavior
        # HINT: The agent will will choose action based on maximum Q-value with
        # '1-ε' probability, and a random action with 'ε' probability.
        if np.random.random() < self.epsilon:
            # With ε probability, choose a random action
            action = self.env.action_space.sample()
        else:
            # With 1-ε probability, choose the action with the highest Q-value
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(get_device())
            with torch.no_grad():
                q_values = self.q_net(obs_tensor)
            action = torch.argmax(q_values).item()
        return action

    def update_q_net(self):
        if len(self.replay_memory.buffer) < self.params['batch_size']:
            return
        # TODO: Update Q-net
        # HINT: You should draw a batch of random samples from the replay buffer
        # and train your Q-net with that sampled batch.
          # Sample a batch of experiences from the replay buffer
        # states, actions, rewards, next_states, not_terminals = self.replay_memory.sample(self.params['batch_size'])

        # states = torch.tensor(np.array(states), dtype=torch.float32).to(get_device())
        # actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(get_device())
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(get_device())
        # next_states = [state for state in next_states if state is not None]
        # not_terminals = torch.tensor(not_terminals, dtype=torch.bool).to(get_device())

        # predicted_state_value = self.q_net(states).gather(1, actions)

        # # Compute target Q-values for the next states
        # with torch.no_grad():
        #     next_state_values = torch.zeros(self.params['batch_size'], device=get_device())
        #     next_state_values[not_terminals] = self.target_net(torch.tensor(np.array(next_states), dtype=torch.float32).to(get_device())).max(1)[0].detach()
        #     target_value = rewards + self.params['gamma'] * next_state_values
        
        s, a, r, ns, not_term = self.replay_memory.sample(self.params['batch_size'])
        s = torch.tensor(np.array(s), dtype=torch.float32).to(get_device())
        a = torch.tensor(a, dtype=torch.long).unsqueeze(1).to(get_device())
        r = torch.tensor(r, dtype=torch.float32).to(get_device())
        not_term = torch.tensor(not_term, dtype=torch.bool).to(get_device())

        # Calculate the predicted state values for the current states and actions using the Q-network
        q_vals = self.q_net(s)
        predicted_s_val = q_vals.gather(1, a)

        # Compute target Q-values for the next states
        with torch.no_grad():
            ns = torch.tensor(np.array([state for state in ns if state is not None]), dtype=torch.float32).to(get_device())
            next_q_vals = torch.zeros(self.params['batch_size'], device=get_device())
            next_q_vals[not_term] = self.target_net(ns).max(1)[0].detach()
            tgt_val = r + self.params['gamma'] * next_q_vals

        # Calculate the Q-network loss
        q_loss = nn.SmoothL1Loss()(predicted_s_val, tgt_val.unsqueeze(1))
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        if len(self.replay_memory.buffer) < self.params['batch_size']:
            return
        q_net_state_dict = self.q_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = self.params['tau']*q_net_state_dict[key] + (1 - self.params['tau'])*target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    def generate_video(self, max_frame=1000):
        self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
        self.epsilon = 0.0
        obs, _ = self.env.reset()
        for _ in range(max_frame):
            action = self.get_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)


class ReplayMemory:
    # TODO: Implement replay buffer
    # HINT: You can use python data structure deque to construct a replay buffer
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, n_samples):
        samples = random.sample(self.buffer, n_samples)
        states, actions, rewards, next_states, not_terminals = zip(*samples)
        return states, actions, rewards, next_states, not_terminals



class QNet(nn.Module):
    # TODO: Define Q-net
    # This is identical to policy network from HW1
    def __init__(self, input_size, output_size, hidden_dim):
        super(QNet, self).__init__()
        self.ff_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, obs):
        return self.ff_net(obs)

