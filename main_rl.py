import argparse, configparser
import os, sys, argparse, collections, pickle, json
import numpy as np
import gym
from util import *
from mapprop import Network

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=False, default="config_cp.ini",
   help="location of config file")
args = ap.parse_args()
f_name = os.path.join("config", "%s" % args.config) 
print("Loading config from %s" % f_name)

config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read(f_name)

name = config.get("USER", "name") # Name of the run
max_eps = config.getint("USER", "max_eps") # Number of episode per run
n_run = config.getint("USER", "n_run") # Number of runs

batch_size = config.getint("USER", "batch_size") # Batch size
env_name = config.get("USER", "env_name") # Environment name
gamma = config.getfloat("USER", "gamma") # Discount rate

hidden = json.loads(config.get("USER","hidden")) # Number of hidden units on each layer
critic_l_type = config.getint("USER", "critic_l_type")  # Activation function for hidden units in critic network; 0 for softplus and 1 for ReLu
actor_l_type = config.getint("USER", "actor_l_type")  # Activation function for hidden units in actor network; 0 for softplus and 1 for ReLu
temp = config.getfloat("USER", "temp") # Temperature for actor network if applicable

critic_var = json.loads(config.get("USER","critic_var")) # Variance in the normal distribution of critic network's layer
critic_update_adj = config.getfloat("USER", "critic_update_adj") # Step size for minimizing the energy of critic network equals to the layer's variance multiplied by this constant
critic_lambda_ = config.getfloat("USER", "critic_lambda_") # Trace decay rate for critic network

actor_var = json.loads(config.get("USER","actor_var")) # Variance in the normal distribution of actor network's layer
actor_update_adj = config.getfloat("USER", "actor_update_adj") # Step size for minimizing the energy of actor network equals to the layer's variance multiplied by this constant
actor_lambda_ = config.getfloat("USER", "actor_lambda_") # Trace decay rate for actor network

map_grad_ascent_steps = config.getint("USER", "map_grad_ascent_steps") # number of step for minimizing the energy
reward_lim = config.getfloat("USER", "reward_lim") # whether limit the size of reward

critic_lr_st = json.loads(config.get("USER","critic_lr_st")) # Learning rate for each critic network's layer at the beginning
critic_lr_end = json.loads(config.get("USER","critic_lr_end")) # Learning rate for each critic network's layer at the beginning
actor_lr_st = json.loads(config.get("USER","actor_lr_st")) # Learning rate for each actor network's layer at the end
actor_lr_end = json.loads(config.get("USER","actor_lr_end")) # Learning rate for each actor network's layer at the end
end_t = config.getint("USER", "end_t") # Number of step to reach the final learning rate (linear interpolation for in-between steps)

# Define constant for activation function
L_SOFTPLUS = 0
L_RELU = 1
L_LINEAR = 2
L_SIGMOID = 3
L_DISCRETE = 4

# Initalize environment
env = batch_envs(name=env_name, batch_size=batch_size, rest_n=0, warm_n=0)   
dis_act = type(env.action_space) != gym.spaces.box.Box

critic_update_size = [i * critic_update_adj for i in critic_var]
actor_update_size = [i * actor_update_adj for i in actor_var]
critic_lambda_ *= gamma
actor_lambda_ *= gamma

print_every = 1000     
eps_ret_hist_full = []

print("Starting experiments on environment %s" % env_name)

for j in range(n_run):          
  critic_net = Network(state_n=env.state.shape[1], action_n=1, hidden=hidden, var=critic_var, 
                       temp=None, hidden_l_type=critic_l_type, output_l_type=L_LINEAR,)   
  output_l_type = L_DISCRETE if dis_act else L_LINEAR   
  action_n = env.action_space.n if dis_act else env.action_space.shape[0]    
  actor_net = Network(state_n=env.state.shape[1], action_n=action_n, hidden=hidden, var=actor_var, 
                     temp=temp, hidden_l_type=actor_l_type, output_l_type=output_l_type)
  
  eps_ret_hist = []  
  c_eps_ret = np.zeros(batch_size)
    
  print_count = print_every         
  value_old = None
  isEnd = env.isEnd
  prev_isEnd = env.isEnd
  truncated, solved, f_perfect = False, False, False
  
  state = env.reset()
  for i in range(int(1e9)):     
    action = actor_net.forward(state)

    if not dis_act:
      action = action * (env.action_space.high[0] - env.action_space.low[0]) - env.action_space.low[0]
      action = np.clip(action, env.action_space.low, env.action_space.high)
    
    value_new = critic_net.forward(state)[:,0]
    mean_value_new = critic_net.layers[-1].mean[:,0]   
    
    if value_old is not None:      
      if reward_lim > 0: reward = np.clip(reward, -reward_lim, +reward_lim)
      targ_value = reward + gamma * mean_value_new * (~isEnd).astype(np.float)      
      critic_reward = targ_value - mean_value_old
      critic_reward[prev_isEnd | info["truncatedEnd"]] = 0          
      actor_reward = targ_value - mean_value_old
      actor_reward[prev_isEnd | info["truncatedEnd"]] = 0            
      
      cur_critic_lr = linear_interpolat(start=critic_lr_st, end=critic_lr_end, end_t=end_t, cur_t=i)
      cur_actor_lr = linear_interpolat(start=actor_lr_st, end=actor_lr_end, end_t=end_t, cur_t=i)  
      
      critic_net.learn(critic_reward, lr=cur_critic_lr)      
      actor_net.learn(actor_reward, lr=cur_actor_lr)          
   
    critic_net.clear_trace(~prev_isEnd)
    critic_net.map_grad_ascent(steps=map_grad_ascent_steps, state=None, gate=True, lambda_=critic_lambda_, 
                    update_size=critic_update_size)           

    actor_net.clear_trace(~prev_isEnd)  
    actor_net.map_grad_ascent(steps=map_grad_ascent_steps, state=None, gate=None, lambda_=actor_lambda_, 
                              update_size=actor_update_size)              

    value_old = np.copy(value_new)
    mean_value_old = np.copy(mean_value_new)
    prev_isEnd = np.copy(isEnd)    
    state, reward, isEnd, info = env.step(from_one_hot(action) if dis_act else action)
      
    c_eps_ret += reward
    
    if np.any(isEnd):
      eps_ret_hist.extend(c_eps_ret[isEnd].tolist())
      c_eps_ret[isEnd] = 0.      
      
    if len(eps_ret_hist) >= max_eps: break       
    
    if i*batch_size > print_count and len(eps_ret_hist) > 0:      
      f_str = "Run %d: Step %d Eps %d\t Running Avg. Return %f\t Max Return %f \t"      
      f_arg = [j+1, i, len(eps_ret_hist), np.average(eps_ret_hist[-100:]), np.amax(eps_ret_hist),]
      print(f_str % tuple(f_arg))
      print_count += print_every          
  eps_ret_hist_full.append(eps_ret_hist)

print("Finished Training")
curves = {}  
curves[name] = (eps_ret_hist_full,)
names = {k:k for k in curves.keys()}
f_name = os.path.join("result", "%s.npy" % name) 
print("Results (saved to %s):" % f_name)
np.save(f_name, curves)
print_stat(curves, names)
plot(curves, names, mv_n=100, end_n=max_eps)

