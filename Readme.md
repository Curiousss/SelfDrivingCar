
#### GameMain.py:
Is the main py file. Here the kivy environment of Assignment 7 was modified to include the T3D:
The car is initialized in a random positions. 
Initially for 1000 time steps the car takes random action to fill the experience replay buffer. For the next 10000 time steps the TD3 is trained. The model is used after 10000 timesteps for maximum 500000 steps.

#### The State of the car using CNN:
The state of the Car in the map is extracted by a CNN model that is pretrained with MNIST dataset using the cropped image of the sand around the car position. Along with the output features of the MNIST layer we add the orientation of the car, the velocity of the car to add more information of the state to TD3.
I suppose resizing the cropped car image to 28x28 does not show much information. Anyway this is the implementation submitted for now.
Alternatively I would want to change the design such that the image of the sand and the car are super imposed and pased through a Network trained to process bigger images. 
