# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
import math
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from .replay_buffer import ReplayBuffer


# detect GPU
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()



class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

steps_done = 0

learnt_rewards = []

def mario_learning(
    env,
    q_func,
    optimizer_spec,
    exploration,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=1000,   
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000
    ):                              #learning_starts = 50000
    
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.MultiDiscrete
    
    
    # 檢查是否是low-dimensional observations (e.g. RAM)
    if len(env.observation_space.shape) == 1:
        input_arg = env.observation_space.shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_arg = frame_history_len * img_c  # 實作論文中的每4 frame擷取一次
    
    
    num_actions = env.action_space.shape

    FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
    
    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            print("DQN output chosen.")
            return model(Variable(obs, volatile=True)).data.max(1)[1].view(1, 1)
        else:
            print("Random output chosen.")
            return torch.IntTensor([[random.randrange(num_actions)]])
        
    # to one hot
    def to_onehot(action, num_actions):
        action = action % num_actions
        if action == 0:
            # Move right while jumping
            action_onehot = np.array([0, 0, 0, 1, 1, 0])
        else:
            action_onehot = np.zeros(num_actions, dtype=int)
            action_onehot[action] = 1
        return action_onehot


    # Initialize target q function and q function
    Q = q_func(input_arg, num_actions).type(dtype)
    target_Q = q_func(input_arg, num_actions).type(dtype)
    
    # Check & load pretrain model
    if os.path.isfile('mario_Q_params.pkl'):
        print('Load Q parametets ...')
        Q.load_state_dict(torch.load('mario_Q_params.pkl'))
        
    if os.path.isfile('mario_target_Q_params.pkl'):
        print('Load target Q parameters ...')
        target_Q.load_state_dict(torch.load('mario_target_Q_params.pkl'))
    
    
    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)
    
    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
    
    ### RUN ENV
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000
    
    
    for t in count(start=steps_done):
        print('timestep:', t)
        ### Step the env and store the transition
        last_idx = replay_buffer.store_frame(last_obs)
        
        recent_observations = replay_buffer.encode_recent_observation()
        
        
        
        if t > learning_starts:
            action = select_epilson_greedy_action(Q, recent_observations, t) [0, 0]
        else:
            action = random.randrange(num_actions)
            
        # one hot encoding
        act_onehot = to_onehot(action, num_actions)

        obs, reward, done, _ = env.step(act_onehot)
        #reward = max(-1.0, min(reward, 1.0))
        replay_buffer.store_effect(last_idx, action, reward, done)
        
        if done:
            obs = env.reset()
        last_obs = obs
        
    
        if (t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size)):
            # print("Optimization starts.")
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
                      

            obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
            act_batch = Variable(torch.from_numpy(act_batch).long())
            rew_batch = Variable(torch.from_numpy(rew_batch))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
            not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(ByteTensor)
            
            if USE_CUDA:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()
                
            
            current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1))
            
            
            _, next_state_actions = Q(next_obs_batch).detach().max(1, keepdim=True)

            next_Q_values = Variable(torch.zeros(batch_size).type(FloatTensor),volatile=False)
            #next_Q_values[not_done_mask] = target_Q(next_obss_batch).max(1)[0]
            next_Q_values[not_done_mask] = target_Q(next_obs_batch).detach().gather(1, next_state_actions)
            

            # TD value
            target_Q_values = rew_batch + (gamma * next_Q_values)
            
            # Compute Bellman error
            Q_bellman_error = F.smooth_l1_loss(current_Q_values, target_Q_values)
            
            
            
            # backward & update
            optimizer.zero_grad()
            Q_bellman_error.backward()

            for param in Q.parameters():
                param.grad.data.clamp_(-1, 1)

            optimizer.step()
            num_param_updates += 1
            
            # target network
            if num_param_updates % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())

            print ("Optimization over.")
                
        ### Log & track
        #
        episode_rewards = env.get_episode_rewards()
        #learnt_rewards = episode_rewards
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:]) # 最近100次reward的平均
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
            
        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            sys.stdout.flush()
            
            # Save the trained model
            torch.save(Q.state_dict(), 'mario_Q_params.pkl')
            torch.save(target_Q.state_dict(), 'mario_target_Q_params.pkl')