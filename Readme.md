# Self Driving car using Kivy and TD3

### GameMain.py:
This is the main py file. Here the kivy environment of Assignment 7 was modified to use the T3D from Assignment 10, to drive the car on the map.
The car is initialized in a random positions. 
Initially for 10000 time steps the car takes random action to fill the experience replay buffer. For the next 10000 time steps the TD3 is trained. The model is used after 10000 timesteps for maximum 500000 steps.

[![Alt text](https://img.youtube.com/vi/SO3KbC9EGHw/0.jpg)](https://www.youtube.com/watch?v=SO3KbC9EGHw)

The car movement is not shown accurately in the output video since the screen recorder app was slowing down the program execution and display.


### The State of the car in CarEnv.py
##### CNN features:
* The the cut out of the sand/map corresponding to the position of the car is one of the inputs to the TD3. It shows the density and shape of the road around the car which can help in deciding the next action. Now the TD3 has CNN layers to get features of this image crop within it.

##### Other State parameters, 7 of them to add more information of the state to TD3:
* The position of the car: x ,y
* The angle of the car,  angle
* The velocity of the car: Velocity xv, yv
* The distance from the goal: xx, yy

##### TD3_NN
Just as earlier the Actor model and Critic model has a couple of Fully connected layers with Relu activation for the hidden layers, and the Actor model also has a Tanh activation at the end. 
Convolutional layers were added to process the image crop of the sand at the car position. Batch normalization was added to all the Convolutional layers and the Fully connected layers except the last one, it helped in getiing regularised output to some extent as opposed to earlier when the output of the Actor was always towards the extreme. 
![alt text](CNN.png)![alt text](Actor.png)![alt text](Critic.png)

Currently this design is not helping the Car agent learn. So this will be changed in the future.

### Action for the car:
The Action is the output of the TD3 model is one value which represent the angle to be taken by the car. The value of the action represents the quantity of the action. Currently the speed on the sand and the road and kept fixed, just as before.
Further I would have wanted to add the speed as another action parameter that should be predicted by the TD3 model.

### Rewards:
The rewards play a key role in how an agent learns using reinforcement. The reward system of Assignment 7 is reused, although I tried different reward systems. But it can definitely be improved. Currently there are different rewards given to different results of the action. 
- A high positive reward of 100 and done-episode set to True for reaching the goal.
- A high negative of -10 reward/done-episode set to True for hitting the wall.
- A positive reward of 1 for staying on the road.
- A negative reward of -1 for staying on the sand.
- A positive reward of 1 for travelling closer to the goal.

The exact value of the reward may not matter as much as the proportion of the reward given to each scenario. It may take an approach like grid search to reach a good reward system.

### Results:
I have been testing only on my laptop which has just 8GB. It was difficult to move to any cloud environment since I found the dependencies of kivy package tricky. So I had to resize the size of the cropped image of the sand to 40x40 from 160x160.
The Car moves properly while taking random actions. Once the TD3 model is used to predict the action sometimes the car spins or flutters. 


[![Alt text](https://img.youtube.com/vi/OeCY7l4m44I/0.jpg)](https://www.youtube.com/watch?v=OeCY7l4m44I)

During training and inference the car sometimes spins, sometimes flutters and sometimes moves coherently.

## Further improvements:
There could be a lot of improvements to the design of what constitutes States, Actions and Rewards, and how the Action predicted is executed in the environment. The Car is not yet learning to stay on roads of reach the goal, so there are some issues with some parts of the code which needs to be corrected.

## Things that were implemented previously:
### Pretrained CNN:
Currently the CNN is added into the Actor and the Critic itself so that the CNN learns along with the other layers of the TD3 which I think is unnecessary burden and the car is not even learning.
Previously I have tried using a pretrained MNIST model it did not make difference since there are some issues with the basic design on States, Actions and Rewards.

### LSTM
I tried to implement LSTMs to capture the movement of the car. It did not make any difference because there is some basic issue with the existing code which needs to be solved before implementing LSTMs. Hence this will be a future plan.

### Open AI gym
Initially I spent a lot of time trying to use open AI gym. I realized that would take more time so I kept it aside and worked with kivy only. I plan to pick up the open gym implementation again in the future.

