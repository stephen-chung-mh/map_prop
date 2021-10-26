import argparse, configparser
import os, sys, argparse, collections, pickle, json
import numpy as np
import gym
from util import *
from mapprop import Network

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=False, default="config_mp.ini",
   help="location of config file")
args = ap.parse_args()
f_name = os.path.join("config", "%s" % args.config) 
print("Loading config from %s" % f_name)

config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read(f_name)

# General parameters
name = config.get("USER", "name") 
max_eps = config.getint("USER", "max_eps") # Number of episode per run
n_run = config.getint("USER", "n_run") # Number of runs

# Task parameters
env_name = config.get("USER", "env_name") # Environment name
batch_size = config.getint("USER", "batch_size") # Batch size

hidden = json.loads(config.get("USER","hidden")) # Number of hidden units on each layer
l_type = config.getint("USER", "l_type")  # Activation function for hidden units in the network; 0 for softplus and 1 for ReLu
temp = config.getfloat("USER", "temp") # Temperature for the network if applicable
var = json.loads(config.get("USER","var")) # variance in hidden layer normal dist.
update_adj = config.getfloat("USER", "update_adj")  # Step size for minimizing the energy of the network equals to the layer's variance multiplied by this constant
map_grad_ascent_steps = config.getint("USER", "map_grad_ascent_steps") # number of step for minimizing the energy
lr = json.loads(config.get("USER","lr")) # Learning rate

# Define constant for activation function
L_SOFTPLUS = 0
L_RELU = 1
L_LINEAR = 2
L_SIGMOID = 3
L_DISCRETE = 4

if env_name == "Multiplexer":
  env=complex_multiplexer_MDP(addr_size=5, action_size=1, 
                            zero=False, reward_zero=False)  
  gate = False
  output_l_type = L_DISCRETE
  action_n = 2**env.action_size

elif env_name == "Regression":
  env=reg_MDP()  
  gate = True
  output_l_type = L_LINEAR
  action_n = 1

update_size = [i * update_adj for i in var]
print_every = 128*500       

eps_ret_hist_full = []
for j in range(n_run):          
  net = Network(state_n=env.x_size, action_n=action_n,  hidden=hidden, var=var, temp=temp, 
               hidden_l_type=l_type, output_l_type=output_l_type)
  
  eps_ret_hist = []
  print_count = print_every         
  for i in range(max_eps//batch_size):  
    state = env.reset(batch_size)        
    action = net.forward(state)
    if env_name == "Multiplexer":
      action = zero_to_neg(from_one_hot(action))[:,np.newaxis]        
      reward = env.act(action)[:,0]    
    elif env_name == "Regression":
      action = action[:,0]   
      reward = env.act(action)
    
    eps_ret_hist.append(np.average(reward))    
    
    net.map_grad_ascent(steps=map_grad_ascent_steps, state=None, gate=gate, lambda_=0, 
                  update_size=update_size) 
    if env_name == "Regression":  reward = env.y - net.layers[-1].mean[:,0]                                 
    net.learn(reward, lr=lr)
      
    if i*batch_size > print_count:
      f_str = "Run %d Step %d Running Avg. Reward \t%f "
      f_arg = [j, i, np.average(eps_ret_hist[-print_every//batch_size:])]
      print(f_str % tuple(f_arg))
      print_count += print_every      
  eps_ret_hist_full.append(eps_ret_hist)

eps_ret_hist_full = np.asarray(eps_ret_hist_full, dtype=np.float32)
print("Finished Training")

curves = {}  
curves[name] = (eps_ret_hist_full,)
names = {k:k for k in curves.keys()}
f_name = os.path.join("result", "%s.npy" % name) 
print("Results (saved to %s):" % f_name)
np.save(f_name, curves)
print_stat(curves, names)
plot(curves, names, mv_n=10, end_n=max_eps)

