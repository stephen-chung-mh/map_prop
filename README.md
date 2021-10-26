# MAP Propagation Algorithm: Faster Learning with a Team of Reinforcement Learning Agents

This repository is the official implementation of the paper "MAP Propagation Algorithm: Faster Learning with a Team of Reinforcement Learning Agents".

## Requirements

Only gym and some basic packages are required to run the code.  To install the requirements, run:

```setup
pip install -r requirements.txt
```

## Training

To train the model on the multiplexer task, run this command:

```train
python main_pre.py -c config_mp.ini
```

To train the model on the scalar regression task, run this command:

```
python main_pre.py -c config_sr.ini
```

To train the model on the Acrobot task, run this command:

```
python main.py -c config_ab.ini
```

To train the model on the CartPole task, run this command:

```
python main.py -c config_cp.ini
```

To train the model on the LunarLander task, run this command:

```
python main.py -c config_ll.ini
```

To train the model on the MountainCar task, run this command:

```
python main.py -c config_mc.ini
```

This will load the config file in `config` folder to run the experiment. By default, 10 runs of training will be done. The result will be stored in the `result` folder and the learning curve will be shown. You can edit the config file to adjust hyperparameters.

## Results

Our model has the following result on the four RL tasks. See paper for details of the result. 

|                                                |    Acrobot     |    CartPole    |  LunarLander   |  MountainCar  |
| :--------------------------------------------- | :------------: | :------------: | :------------: | :-----------: |
| Average return over all episodes - Mean (Std.) | -100.29 (5.40) | 459.70 (13.89) | 127.88 (24.57) | 39.45 (30.48) |

## Contributing

This software is licensed under the Apache License, version 2 ("ALv2").