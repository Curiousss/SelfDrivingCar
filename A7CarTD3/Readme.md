
The kivy environment was modified to include the T3D. The car is initialized in a random position. Initially for 1000 time steps the car takes random action to fill the experience replay buffer. For the next 10000 time steps the TD3 is trained. The model is used after 10000 timesteps for maximum 500000 steps.

The Car state is the features extracted by a CNN model that is pretrained with MNIST dataset.
