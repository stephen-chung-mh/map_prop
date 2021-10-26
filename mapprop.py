import numpy as np 
from util import *

L_SOFTPLUS = 0
L_RELU = 1
L_LINEAR = 2
L_SIGMOID = 3
L_DISCRETE = 4
LS_REAL = [L_SOFTPLUS, L_RELU, L_LINEAR, L_SIGMOID]

ACT_F = {L_SOFTPLUS: softplus,
         L_RELU: relu,
         L_SIGMOID: sigmoid,
         L_LINEAR: lambda x: x,
         }

ACT_D_F = {L_SOFTPLUS: sigmoid,
           L_RELU: relu_d,
           L_SIGMOID: sigmoid_d,
           L_LINEAR: lambda x: 1,
           }

class eq_prop_layer():
  def __init__(self, name, input_size, output_size, optimizer, var, temp, l_type):
    if l_type not in [L_SOFTPLUS, L_RELU, L_LINEAR, L_SIGMOID, L_DISCRETE]:
      raise Exception('l_type (%d) not implemented' % l_type)
    
    self.name = name
    self.input_size = input_size
    self.output_size = output_size
    self.optimizer = optimizer
    self.l_type = l_type
    self.temp = temp if l_type == L_DISCRETE else 1
    
    lim = np.sqrt(6 / (input_size + output_size))       
    self._w = np.random.uniform(-lim, lim, size=(input_size, output_size))
    self._b = np.zeros(shape=output_size)    
    self._inv_var = np.full(output_size, 1/var) 
    
    self.prev_layer = None # Set manually
    self.next_layer = None # Set manually
    self.values = np.zeros((1, output_size))
    self.new_values = np.zeros((1, output_size))

    self.w_trace = np.zeros((1, input_size, output_size,))
    self.b_trace = np.zeros((1, output_size,))    
    
  def sample(self, inputs):    
    self.compute_pot_mean(inputs)          
    if self.l_type in LS_REAL:      
      sigma = np.sqrt(1/self._inv_var)
      self.values = self.mean + sigma * np.random.normal(size=self.pot.shape)
      return self.values
    elif self.l_type == L_DISCRETE:      
      self.values = multinomial_rvs(n=1, p=self.mean)
      return self.values
  
  def compute_pot_mean(self, inputs):
    # Compute potential (pre-activated mean value) and mean value of layer
    self.inputs = inputs    
    self.pot = (inputs.dot(self._w) + self._b)
    if self.l_type in LS_REAL:        
      self.mean = ACT_F[self.l_type](self.pot)
    else:       
      self.mean = softmax(self.pot/self.temp, axis=-1)
  
  def refresh(self, freeze_value):    
    if self.prev_layer is not None: self.inputs = self.prev_layer.new_values                
    self.compute_pot_mean(self.inputs)   
    if not freeze_value: self.values = self.new_values    
  
  def update(self, update_size,):
    # MAP est. gradient ascent, assuming all layers are refreshed 
    if self.next_layer is None: 
      if self.l_type in LS_REAL:
        sigma = np.sqrt(1/self._inv_var)
        self.new_values = self.mean + sigma * np.random.normal(size=self.pot.shape)      
      elif self.l_type == L_DISCRETE:      
        self.new_values = multinomial_rvs(n=1, p=self.mean)
      #self.new_values = self.mean
    elif self.l_type in LS_REAL:
      lower_pot = (self.mean - self.values) * self._inv_var
      if self.next_layer is None:
        upper_pot = 0.
      else:
        fb_w = self.next_layer._w.T          
        if self.next_layer.l_type in LS_REAL:
          upper_pot = ((self.next_layer.values - self.next_layer.mean) *
                      ACT_D_F[self.next_layer.l_type](self.next_layer.pot) * self.next_layer._inv_var).dot(fb_w)      
        else:
          upper_pot = (self.next_layer.values - self.next_layer.mean).dot(fb_w)/self.next_layer.temp           
        
      update_pot = lower_pot + upper_pot        
      update_step = update_size * update_pot 
      self.new_values = self.values + update_step    

  def record_trace(self, gate=None, lambda_=0):
    if self.l_type in LS_REAL:
      v_ch = (self.values - self.mean) * ACT_D_F[self.l_type](self.pot) * self._inv_var
    else:
      v_ch = (self.values - self.mean) / self.temp
    if gate is not None:       
      v_ch *= gate[:, np.newaxis]
      
    self.w_trace *= lambda_
    self.b_trace *= lambda_            
    self.w_trace = self.w_trace + self.inputs[:, :, np.newaxis] * v_ch[:, np.newaxis, :] 
    self.b_trace = self.b_trace + v_ch       
      
  def learn_trace(self, reward, lr=0.01):        
    w_update = self.w_trace * reward[:, np.newaxis, np.newaxis]   
    b_update = self.b_trace * reward[:, np.newaxis]

    w_update = np.average(w_update, axis=0)   
    b_update = np.average(b_update, axis=0) 
    
    delta_w = self.optimizer.delta(grads=[w_update], name=self.name+"_w", learning_rate=lr)[0]    
    delta_b = self.optimizer.delta(grads=[b_update], name=self.name+"_b", learning_rate=lr)[0]    
    self._w += delta_w    
    self._b += delta_b
      
  def clear_trace(self, mask):    
    self.w_trace = self.w_trace * (mask.astype(np.float))[:, np.newaxis, np.newaxis]
    self.b_trace = self.b_trace * (mask.astype(np.float))[:, np.newaxis]
 
  def clear_values(self, mask):    
    self.values = self.values * (mask.astype(self.values.dtype))[:, np.newaxis]
    self.new_values = self.new_values * (mask.astype(self.new_values.dtype))[:, np.newaxis]
    if self.mv_adj: self.mv[mask==0] = 1/self.mv_alpha
      
class Network():
  def __init__(self, state_n, action_n, hidden, var, temp, 
               hidden_l_type, output_l_type,):
    
    self.layers = []    
    in_size = state_n
    optimizer = adam_optimizer(learning_rate=0.01, beta_1=0.9, beta_2=0.999)           

    for d, n in enumerate(hidden + [action_n]):
      a = eq_prop_layer(name="layer_%d"%d, input_size=in_size, output_size=n, 
                        optimizer=optimizer, var=getl(var,d),  temp=temp,
                        l_type=(output_l_type if d==len(hidden) else hidden_l_type),)
      if d > 0: 
        a.prev_layer = self.layers[-1]        
        self.layers[-1].next_layer = a 
      self.layers.append(a)                  
      in_size = n

  def forward(self, state):        
    self.state = state
    h = state   
    for n, a in enumerate(self.layers): h = a.sample(h)                    
    self.action = h                
    return self.action        
  
  def map_grad_ascent(self, steps, state=None, gate=False, lambda_=0, update_size=0.01):
    for i in range(steps):              
      for n, a in enumerate(self.layers[:-1] if state is None else self.layers):
        a.update(update_size=getl(update_size,n))              
      for n, a in enumerate(self.layers): 
        a.refresh(freeze_value = (n == (len(self.layers)-1) and state is None))         
      if i == steps-1:
        gate_c = 1/(self.layers[-1].values - self.layers[-1].mean)[:,0] if gate else None
        for n, a in enumerate(self.layers): a.record_trace(gate=gate_c, lambda_=lambda_)
          
  def learn(self, reward, lr=0.01):    
    for n, a in enumerate(self.layers):
      a.learn_trace(reward=reward, lr=getl(lr, n))
          
  def clear_trace(self, mask):
    for n, a in enumerate(self.layers): a.clear_trace(mask)
      
  def clear_values(self, mask):
    for n, a in enumerate(self.layers): a.clear_values(mask)      