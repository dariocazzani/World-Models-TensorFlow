## An Implementation of [World Models](https://worldmodels.github.io/) in TensorFlow

[![dep2](https://img.shields.io/badge/TensorFlow-1.3%2B-orange.svg)](https://www.tensorflow.org/)
![dep1](https://img.shields.io/badge/Status-Work--in--Progress-brightgreen.svg)
[![dep2](https://img.shields.io/badge/OpenAI-Gym-blue.svg)](https://gym.openai.com/)

![car-driving](https://github.com/dariocazzani/World-Models-TensorFlow/blob/master/images/car-drive-dream.gif)

### Create a Python Virtual Environment

```
mkvirtualenv --python=/usr/bin/python3 World-Models-TensorFlow
```

###  Install dependencies
```
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

pip install -r requirements.txt
```

### 1. Stateless Agents

Train the agents to drive around using only the information from the current frame.

```
cd 01-VAE/
```
And follow the instructions there

### 2. Train the RNN

Work in Progress
