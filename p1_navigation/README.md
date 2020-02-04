[//]: # (Image References)

# Udacity Deep Reinforcement Learning Nano Degree: Project 1 - Navigation

## Introduction

I trained an agent to navigate in a large, square world with bananas.

<img src="images/my-trained-agent.gif" width="60%" align="top-left" alt="" title="my trained agent" />

*above: my trained agent*

The task is episodic. The agent gets a reward of +1 for collecting a yellow banana and a reward of -1 for collecting a blue banana. During one episode the agent is to collect as many yellow bananas as possible while avoiding blue bananas. Every episode is 300 steps.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The goal is to train the agent to get an average score of +13 over 100 consecutive episodes.

## Getting Started

### Set up your python environment

1. Create and activate a new environment with Python 3.6.

	- __Linux__ or __Mac__:
	```bash
	conda create --name drl-navigation python=3.6
	source activate drl-navigation
	```
	- __Windows__:
	```bash
	conda create --name drl-navigation python=3.6
	activate drl-navigation
	```

2. Clone this repository and install the dependencies.
```bash
git clone https://github.com/AnKra/udacity-deep-reinforcement-learning-navigation-project.git
cd udacity-deep-reinforcement-learning-navigation-project/python
pip install .
cd ..
```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drl-navigation` environment.  
```bash
python -m ipykernel install --user --name drl-navigation --display-name "drl-navigation"
```

### Set up the banana environment

1. Choose the environment matching your operating system from one of the links below.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the zip file your working copy of this repository and unzip it.

### Run the notebook

1. Open a notebook
```bash
jupyter notebook
```

2. Before running code in a notebook, change the kernel to match the `drl-navigation` environment by using the drop-down `Kernel` menu.

3. Follow the instructions in `Navigation.ipynb`
