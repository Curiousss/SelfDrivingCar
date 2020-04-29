'''

    # Self Driving Car by Meera - Implementing TD3 with the custom Car Env
    @author = "Meera"
    @copyright = "Copyright 2020, The Self Driving Car Project RL Custom Environment with TD3"
    @license = "GPL"
    @version = "1.0.1"
    @maintainer = "Sess102020"
'''

# Importing the libraries
import numpy as np
import random

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
from CarWidget import Car as car

class CarEnv:
    # class CarEnv():
    '''

        # Self Driving Car by Meera - Implementing TD3 with the custom Car Env
        @author = "Meera"
        @copyright = "Copyright 2020, The Self Driving Car Project RL Custom Environment with TD3"
        @license = "GPL"
        @version = "1.0.1"
        @maintainer = "Sess102020"
    '''

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 35}

    #Sess10: TD3 Variable declaration by Sess10 STARTS
    _max_episode_steps = 1000 # Max number of steps in a single episode

    #State Dimension
    # state_dim = 10002

    # Car Steering wheel has one degree of freedom
    # angle -1 to 1, velocity 0 to 1
    low = -1  # Equivalent of -180 degree . low = np.array([-180])
    high = 1  # Equivalent of +180 degree. This is also the value of max_action
    max_rotation_angle = 20  #Max can be 180 degrees
    # Sess10: TD3 Variable declaration by Sess10 ENDS

    # Size of the area to be cropped around the Car
    pixel_crop_size = (160, 160) #Note: To be give in even numbers
    # env_width = 975
    # env_height = 548
    #Sess10: Method added by Sess10

    def __init__(self):
        return
    #    super(CarEnv, self).__init__()

    # TOdo Meera Implementing gy, env.sample here.
    def sample():
        return [random.uniform(-1,1)]

    #NOTE: self is the context of the calling class which is the Game class here
    def reset(self):
        '''
                Initialization of the environment related variables
                These would need to be declared 'global' again in 'Game.update' function to be accessible in the 'Game.update' function
                I think one reason to declare these variables global so that they are available across the classes of Kivy
                Other reason could be to have singleton pattern
        '''
        # START OF THE LIST OF VARIABLES DECLARATION---------------
        global last_reward
        global last_distance
        global action2rotation

        global longueur
        global largeur

        global im # a variable specifically to read the pixels from the env
        global sand

        global goal_x
        global goal_y
        global swap

        global first_update

        # END OF LIST OF VARIABLES DECLARATION---------------

        # Intializing the Environment variables
        # scores = [] #This is used to Track the cumulative score to exit the game. This can also be taken out of the environment class

        last_reward = 0
        last_distance = 0
        # action2rotation = [0, 5, -5]  # Since, this is CONSTANT, this taken out from reset method
        # action2rotation = [0, 10, -10]  # Since, this is CONSTANT, this taken out from reset method
        # textureMask = CoreImage(source="./kivytest/simplemask1.png")
        im = CoreImage("./images/MASK1.png")

        # Attributes of of Game(Widget) class. Can be Hardcoded or Passed as an arg to reset() method
        longueur = self.width
        largeur = self.height

        # longueur = self.env_width
        # largeur = self.env_height

        sand = np.zeros((longueur, largeur))
        img = PILImage.open("./images/mask.png").convert('L')  # Else gives error cannot write mode F as JPEG for grayscale image if try to save
        # img = PILImage.open("./images/mask.png")
        sand = np.asarray(img) / 255

        goal_x = 430
        goal_y = 323  # 548 - 262

        # '975') # 100 --875
        # '548') # 100 -- 448
        init_x = np.random.randint(100, 675)
        init_y = np.random.randint(100, 440)
        self.car.x = init_x
        self.car.y = init_y

        # kivy file parameters
        # < Goal1 >: 892, 300
        # < Goal2 >: 430, 323
        # < Goal3 >: 105, 397

        # NOTE: self is the context of the calling class which is the Game class here
        # print('self.goal1.x :', self.goal1.x)  #   self.goal1.x : 477.5
        # print('self.goal1.y :', self.goal1.y)  #   self.goal1.y : 264
        # print('self.width :', self.width)  #   975
        # print('self.height :', self.height)  #   548

        swap = 0
        first_update = True

        #Gathering the parameters for Training
        last_reward, episode_over = CarEnv.get_reward(self)
        pixel_state = CarEnv.get_state(self) #last_signal is the state
        episode_over = False

        return pixel_state, last_reward, episode_over, {}

    def get_goal(self):

        global goal_x
        global goal_y

        # ADDITIONAL PARAMETER TO BE PASSED TO NN
        # goal_x, goal_y = env.get_goal(self)
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y

        # orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
        orientation = Vector(*self.car.velocity).angle((goal_x, goal_y)) / 180.

        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)

        # self.car.velocity = Vector(2, 0).rotate(self.car.angle)

        return goal_x, goal_y, orientation, distance, self.car.velocity, self.car.angle

    def get_reward(self):
        # Accessing these gloabl variables (the below declaration is mandatory for the same)

        global brain
        global scores

        global last_reward
        global last_distance
        global action2rotation

        global longueur
        global largeur

        global im  # a variable specifically to read the pixels from the env
        global sand

        global goal_x
        global goal_y
        global swap

        global first_update

        global episode_over

        # Sess10: Reward Calculations based on the action taken

        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        # If the car's center is in sand, give a negative reward
        if sand[int(self.car.x), int(self.car.y)] > 0:

            # Sess10: increased the velocity in sand in order to explore more
            # self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            # print(1, goal_x, goal_y, distance, int(self.car.x), int(self.car.y),
            #       im.read_pixel(int(self.car.x), int(self.car.y)))

            goal_tuple =(goal_x, goal_y)
            car_coordinate_tuple = (int(self.car.x), int(self.car.y))
            print("position after action:: {}  goal : {} distance to goal: {} car location: {} map_pixels: {}"
                  .format('1(on sand)', goal_tuple, round(distance,2), car_coordinate_tuple, im.read_pixel(int(self.car.x), int(self.car.y)))
                  )

            # Sess10: Uncommented
            last_reward = -1 #Make a negative reward

        else:  # otherwise
            # self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            # self.car.velocity = Vector(4, 0).rotate(self.car.angle)
            #Sess10: reversing the velocity of the Sand vs Road inorder to increase the Road Transitions
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward = 1  # Make a positive reward
            # last_reward = 8

            # print(0, goal_x, goal_y, distance, int(self.car.x), int(self.car.y),
            #       im.read_pixel(int(self.car.x), int(self.car.y)))

            goal_tuple =(goal_x, goal_y)
            car_coordinate_tuple = (int(self.car.x), int(self.car.y))
            print("position after action:: {}  goal : {} distance to goal: {} car location: {} map_pixels: {}"
                  .format('0(on road)', goal_tuple, round(distance,2), car_coordinate_tuple, im.read_pixel(int(self.car.x), int(self.car.y)))
                  )

            # if distance < self.last_distance:
            if distance < last_distance:
                last_reward = 1 #Todo Meera
                # last_reward += 10
                # last_reward += 0.1
            # else:
            #     last_reward = last_reward +(-0.2)
                # last_reward -= 0.1

        last_distance = distance

        rows = CarEnv.pixel_crop_size[0] #100
        columns = CarEnv.pixel_crop_size[1] #100
        allowed_width_from_boundry = (rows//2) + 5 #adding two pixels extra else crop will fail depending on velocity etc

        # if self.car.x < 55:
        if self.car.x < allowed_width_from_boundry:
            self.car.x = allowed_width_from_boundry
            # last_reward += -20
            episode_over = True
            last_reward = -10

        if self.car.x > self.width - allowed_width_from_boundry:
            self.car.x = self.width - allowed_width_from_boundry
            # last_reward += -20
            episode_over = True
            last_reward = -10

        if self.car.y < allowed_width_from_boundry:
            self.car.y = allowed_width_from_boundry
            # last_reward += -20
            episode_over = True
            last_reward = -10

        if self.car.y > self.height - allowed_width_from_boundry:
            self.car.y = self.height - allowed_width_from_boundry
            # last_reward += -20
            episode_over = True
            last_reward = -10

        # Giving a reward on reaching the goal. Note here, the goal has been reset instead of returning done = True

        episode_over = False

        if distance < 40:
            last_reward = 100  # Added by Sess10
            # last_reward += 100  # Added by Sess10
            episode_over = True # Added by Sess10

            #     done = True

            if swap == 0:
                goal_x = 892
                goal_y = 300
                swap = 1
                # last_reward = 5

            elif swap == 1:
                goal_x = 430
                goal_y = 323
                # swap = -1
                swap = 2
                # last_reward = 10

            else:
                goal_x = 105
                goal_y = 397
                swap = 0
                # last_reward = 5

        # print("Sess10 CarEnv last_reward: type(last_reward) is : ", type(last_reward))
        # print("Sess10 CarEnv last_reward: type(last_reward) is : ", last_reward)

        return last_reward, episode_over

    def take_action(self, action):
        # This takes the action as input, performs the action and returns the reward
        print("action", action)

        # rotation = action2rotation[action]  # Actually, this rotation is the action, I can do this inside step method also

        # type action value is: <class 'numpy.ndarray'> with float value between -1 and  +1 e.g. 119.7109580039978
        rotation = action[0] * CarEnv.max_rotation_angle  # max_rotation_angle = 180 degrees
        #TOdo: Meera Take this rotation = float(rotation) # Rounding it to one decimal place. TBD: Float values are not working somehow

        # Sess10: Below six lines are helping the Car to move  -- angle and
        self.car.pos = Vector(*self.car.velocity) + self.car.pos
        self.car.rotation = float(rotation)
        self.car.angle = (self.car.angle + self.car.rotation) % 360# Sess10: This needs to be corrected. Modified 18th Apr
        #print("ROtation", rotation)
        print("Angle", self.car.angle)


        # self.car.sensor1 = Vector(30, 0).rotate(self.car.angle) + self.car.pos
        # self.car.sensor2 = Vector(30, 0).rotate((self.car.angle + 30) % 360) + self.car.pos
        # self.car.sensor3 = Vector(30, 0).rotate((self.car.angle - 30) % 360) + self.car.pos

        global last_reward
        last_reward, episode_over = CarEnv.get_reward(self)  # Reward got after taking the action

        return last_reward, episode_over

    def step(self, action):

        global episode_over

        # return ob, reward, episode_over, {}  #ob, reward, episode_over, info : tuple
        old_action = action
        # action = [0] * NUM_ACTIONS  #Not required
        # Let it also accept a list of actions and put the same validation logic as in DOOM class
        last_reward, episode_over = CarEnv.take_action(self, action)  # This method should return the last_reward. distance is not required as reward is based on distance
        # last_reward = self.car.get_reward()  #This step is not required as getting the reward from take action itself
        # state, pixel_state = CarEnv.get_state(self)  # gives the current state of the env and agent
        pixel_state = CarEnv.get_state(self)  # gives the current state of the env and agent
        # Sess10: Make it same as self.game.get_state()
        # episode_over = False
        # print("Sess10 CarEnv step: type(last_reward) is : ", type(last_reward))
        # print("Sess10 CarEnv step: The last_reward is : ", last_reward)
        # print("Sess10 CarEnv step: The episode_over is : ", episode_over)
        # return state, last_reward, episode_over, {"pixel_state": pixel_state}
        return pixel_state, last_reward, episode_over, {}

    def get_state(self):
        # Sess10: The below three lines are returning the current state of the CAR (20x20 square is getting cropped)
        # self.car.signal1 = int(np.sum(sand[int(self.car.sensor1_x) - 10:int(self.car.sensor1_x) + 10,
        #                               int(self.car.sensor1_y) - 10:int(self.car.sensor1_y) + 10])) / 400.
        # self.car.signal2 = int(np.sum(sand[int(self.car.sensor2_x) - 10:int(self.car.sensor2_x) + 10,
        #                               int(self.car.sensor2_y) - 10:int(self.car.sensor2_y) + 10])) / 400.
        # self.car.signal3 = int(np.sum(sand[int(self.car.sensor3_x) - 10:int(self.car.sensor3_x) + 10,
        #                               int(self.car.sensor3_y) - 10:int(self.car.sensor3_y) + 10])) / 400.
        # Sess10: Forget about the below lines as of now
        # if self.car.sensor1_x > longueur - 10 or self.car.sensor1_x < 10 or self.car.sensor1_y > largeur - 10 or self.car.sensor1_y < 10:
        #     self.car.signal1 = 10.
        # if self.car.sensor2_x > longueur - 10 or self.car.sensor2_x < 10 or self.car.sensor2_y > largeur - 10 or self.car.sensor2_y < 10:
        #     self.car.signal2 = 10.
        # if self.car.sensor3_x > longueur - 10 or self.car.sensor3_x < 10 or self.car.sensor3_y > largeur - 10 or self.car.sensor3_y < 10:
        #     self.car.signal3 = 10.

        global goal_x
        global goal_y

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.

        # Todo: Meera
        # last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation] # state
        last_signal = [self.car.x, self.car.y, xx, yy, *self.car.velocity, self.car.angle] # state

        # crop area 20x20
        rows = CarEnv.pixel_crop_size[0]
        columns = CarEnv.pixel_crop_size[1]

        pixel_state = sand [int(self.car.x) - rows//2:int(self.car.x) + rows//2, int(self.car.y) - columns//2:int(self.car.y) + columns//2]
        # pixel_state = sand [int(self.car.x) - 10:int(self.car.x) + 10, int(self.car.y) - 10:int(self.car.y) + 10]
        # Need not worry about map boundry scenario as car is always kept 50 pixels away from the boundry in the get_reward function
        pixel_state = np.resize(pixel_state, (1, 40, 40))
        #print("pixel_state",pixel_state)
        norm = np.linalg.norm(last_signal)
        last_signal = last_signal / norm
        print("last_signal", last_signal)

        full_state = [pixel_state, last_signal]
        return full_state

