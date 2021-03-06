# RL with Atari

Adapted from CS234 Assignment 2

# Current Implementation Details
* 09-22-21: q4_nature_torch runs on toy example but achieves maximum reward of 1, this is potentially buggy (see results/q4_nature/scores.png)
* Versus q3_linear which achieves maximum reward of 6 (corresponding to optimal play)

## Environment

### Pong-v0

- We play against a decent AI player.
- One player wins if the ball pass through the other player and gets reward +1 else -1.
- Episode is over when one of the player reaches 21 wins
- final score is between -21 or +21 (lost all or won all)

```python
# action = int in [0, 6)
# state  = (210, 160, 3) array
# reward = 0 during the game, 1 if we win, -1 else
```

We use a modified env where the dimension of the input is reduced to

```python
# state = (80, 80, 1)
```

with downsampling and greyscale.

## Training

Once done with implementing `q2_linear.py` (setup of the tensorflow necessary op) and `q3_nature` make sure you test your implementation by launching `python q2_linear.py` and `python q3_nature.py` that will run your code on the Test environment.

You can launch the training of DeepMind's DQN on pong with

```
python q5_train_atari_nature.py
```

The default config file should be sufficient to reach good performance after 5 million steps.



Training tips: 
(1) The starter code writes summaries of a bunch of useful variables that can help you monitor the training process.
You can monitor your training with Tensorboard by doing, on Azure

```
tensorboard --logdir=results
```

and then connect to `ip-of-you-machine:6006`


(2) You can use ‘screen’ to manage windows on VM and to retrieve running programs. 
Before training DQN on Atari games, run 

```
screen 
```
then run 

```
python q5_train_atari_nature.py
```
By using Screen, programs continue to run when their window is currently not visible and even when the whole screen session is detached 
from the users terminal. 

To detach from your window, simply press the following sequence of buttons

```
ctrl-a d
```
This is done by pressing control-a first, releasing it, and press d


To retrieve your running program on VM, simply type

```
screen -r
```
which will recover the detached window.   



**Credits**
Assignment starter code written by Guillaume Genthial and Shuhui Qu.
Assignment starter code updated by Haojun Li and Garrett Thomas
