## Train Stateless Agents

### 1. Generate Data for the Variational AutoEncoder
```
mkdir -p ../data
python generate-VAE-data.py
```

### 2. Train the Variational AutoEncoder
```
python train_VAE.py
```

In another terminal run the following command to visualize the training in TensorBoard
```
tensorboard --logdir logdir
```

![tensorboard-VAE](https://github.com/dariocazzani/World-Models-TensorFlow/blob/master/images/tensorboard-VAE.png)

### 3. Train the agents
```
python train-agents.py
```

In another terminal run the following command to visualize the training
```
python display_rewards.py
```
