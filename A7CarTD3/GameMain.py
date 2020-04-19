'''

    # Self Driving Car by Meera - Implementing TD3 with the custom Car Env
    @author = "Meera"
    @copyright = "Copyright 2020, The Self Driving Car Project RL Custom Environment with TD3"
    @license = "GPL"
    @version = "1.0.1"
    @maintainer = "Sess102020"
'''

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

#TD3 Related imports
import os
import time
# import random
import numpy as np
# import matplotlib.pyplot as plt
# import pybullet_envs
# import gym
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from gym import wrappers
# from torch.autograd import Variable
# from collections import deque

#Importing TD3 classes and Feature Extractor
from TD3_ExperienceReplay import ReplayBuffer
from TD3 import TD3
#from FE_CNN_DCQN import DCQN_FeatureExtractor
from FE_CNN_MNIST import MNIST_FeatureExtractor

# Importing the Custom Environment
from CarEnv import CarEnv as env
from CarWidget import Car, Ball1, Ball2, Ball3, Goal1, Goal2, Goal3

from pathlib import Path
from PIL import Image

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse, multitouch_on_demand')
Config.set('graphics', 'resizable', False)

Config.set('graphics', 'width', '975')
Config.set('graphics', 'height', '548')

# Initialising the necessary parameters. Modify these for experimentation
pixel_crop_size = env.pixel_crop_size #(160, 160)
no_stacked_frame = 4
input_size = (no_stacked_frame, pixel_crop_size[0], pixel_crop_size[1], ) # Input to the PyTorch Feature Extractor... PyTorch takes channel first format (4, 100, 100)

# Feature Extracted Related Parameters
# state_dimension = 45  # 1D array of getTransformedStateVector method
state_dimension = 205  # 1D array of getTransformedStateVector method
# feature_extraction_mode = 'cnn_dcqn' # Valid modes: 'frame_diff', 'cnn_dcqn', 'cnn_mnist'
feature_extraction_mode = 'cnn_mnist' # Valid modes: 'frame_diff', 'cnn_dcqn', 'cnn_mnist'
# feature_extrator_model_name='./feature_extraction/last_brain_54000.pth'
feature_extrator_model_name='./feature_extraction_MNIST/MNIST_model.pth'

# Training Related
train_n_steps = 500 #Train the model every n steps
first_update = True

#delete it
save_state_image = False

# newstate_images_path
# currentstate_images_path

def save_image(stateobs, counter):

    #Note: these saved images can be matched with the total_timestamp value which is getting printed on consloe
    state_images_path = Path(
        r"D:\PyCharmProjects\FaceRecognition_Assignment_TSAI\REINFORCEMENT_LEARNING\EVA_Assignment_P2S10\CarEnv_DCQN_WIP_NJ\TD3_rewriting\saved_states")

    new_state_img = Image.fromarray(np.uint8(stateobs * 255))
    new_state_img = new_state_img.convert("L")  # To solve error in .save: cannot write mode F as JPEG
    img_file_name = str(counter) + ".jpg"
    new_state_img.save(state_images_path / img_file_name)


class Game(Widget):


    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    goal1 = ObjectProperty(None)
    goal2 = ObjectProperty(None)
    goal3 = ObjectProperty(None)

    def serve_car(self):

        # '975') # 100 --875
        # '548') # 100 -- 448
        # init_x = np.random.randint(10, 100)
        # init_y = np.random.randint(10, 100)

        self.car.center = self.center
        # self.car.center = (init_x, init_y)
        # self.car.x = init_x
        # self.car.y = init_y
        self.car.velocity = Vector(6, 0)

    # Note that the First method after serve_car getting called is the 'Game.update'
    def train(self, dt):

        from CarEnv import CarEnv as env

        #Accessing these gloabl variables (the below declaration is mandatory for the same)

        # global brain
        # global scores

        global last_reward
        # global last_distance
        # global action2rotation

        global longueur
        global largeur

        global im  # a variable specifically to read the pixels from the env
        global sand

        global goal_x
        global goal_y
        global swap

        global first_update

        # global epsilon
        # global epsilonDecayRate
        # global minEpsilon

        # longueur = self.width
        # largeur = self.height

        global currentState
        global nextState

        #Sess10: TD3 parameters initialization for Kivy: STARTS
        global total_timesteps
        global max_timesteps
        global done
        global start_timesteps
        global episode_timesteps
        global episode_reward
        global replay_buffer
        global obs
        global timesteps_since_eval

        global CURRENT_1D_STATE_TO_BE_PASSED_TO_NN
        global NEXT_1D_STATE_TO_BE_PASSED_TO_NN
        global policy
        global expl_noise
        global episode_num
        global batch_size, discount, tau, policy_noise, noise_clip, policy_freq
        global eval_freq
        global file_name

        global state_dimension #size of the 1D array to be passed to Actor-Critic models
        # TD3 parameters initialization for Kivy: ENDS

        global epsilon
        global epsilonDecayRate
        global minEpsilon


        # print('first update parameter is', first_update)
        if first_update:
            last_signal, last_reward, episode_over, info = env.reset(self)  #Note: env.reset is not returning anything here as state is getting initialised in the Game Widget

            print("In 'first_update': last_reward value is : ", last_reward)

            currentState, nextState = resetPixelStates(self)
            obs = env.get_state(self)

            # Epsilon greedy Algorithm
            # epsilon = 1.
            epsilon = 0.05
            epsilonDecayRate = 0.0002
            minEpsilon = 0.05

            # first_update = False

        if first_update:

            # ## We set the parameters
            # env_name = "HalfCheetahBulletEnv-v0"  # Name of a environment (set it to any Continous environment you want)
            env_name = "CarEnv-v0"  # Name of a environment (set it to any Continous environment you want)
            seed = 0  # Random seed number

            #Sess10: Manupulating this to debug
            start_timesteps = 1e4  # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
            # start_timesteps = 10

            eval_freq = 5e3  # How often the evaluation step is performed (after how many timesteps)
            # max_timesteps = 5e5  # Total number of iterations/timesteps
            max_timesteps = 5e6  #Sess10: Inscreasing the number of training steps
            save_models = True  # Boolean checker whether or not to save the pre-trained model
            # expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
            expl_noise = 0.2 #Sess10: Modified by Sess10

            batch_size = 100  # Size of the batch
            # batch_size = 256  # Size of the batch

            discount = 0.99  # Discount factor gamma, used in the calculation of the total discounted reward
            tau = 0.005  # Target network update rate
            policy_noise = 0.2  # STD of Gaussian noise added to the actions for the exploration purposes
            noise_clip = 0.5  # Maximum value of the Gaussian noise added to the actions (policy)
            policy_freq = 2  # Number of iterations to wait before the policy network (Actor model) is updated

            # ## We create a file name for the two saved models: the Actor and Critic models

            file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
            print("---------------------------------------")
            print("Settings: %s" % (file_name))
            print("---------------------------------------")

            # ## We create a folder inside which will be saved the trained models

            if not os.path.exists("./results"):
                os.makedirs("./results")
            if save_models and not os.path.exists("./pytorch_models"):
                os.makedirs("./pytorch_models")

            # ## We create the PyBullet environment
            # MAKING THE ENVIRONMENT AND GETTING THE ENV OBJECT
            # Sess10: TBD : CarEnv has been imported.....How to make it in the below way
            # env = gym.make(env_name)

            # ## We set seeds and we get the necessary information on the states and actions in the chosen environment
            torch.manual_seed(seed)
            np.random.seed(seed)

            # env.seed(seed)
            # state_dim = env.observation_space.shape[0]
            state_dim = state_dimension
            action_dim = env.action_space.shape[0]  # 1
            max_action = float(env.action_space.high[0]) # 1.0

            #Sess10: TBD: Make them part of the CarEnv class
            # state_dim = 10002  # Size of the 1D array to be passed

            # action_dim = 1
            # max_action = 1.0

            # ## We create the policy network (the Actor model)
            policy = TD3(state_dim, action_dim, max_action)

            # ## We create the Experience Replay memory
            replay_buffer = ReplayBuffer()

            # ## We define a list where all the evaluation results over 10 episodes are stored
            # Sess10: Temporary commented the evaluations
            # evaluations = [evaluate_policy(self, env, policy)]

            work_dir = mkdir('exp', 'brs')
            monitor_dir = mkdir(work_dir, 'monitor')
            max_episode_steps = env._max_episode_steps
            save_env_vid = False

            if save_env_vid:
                env = wrappers.Monitor(env, monitor_dir, force=True)
                env.reset()

            # ## We initialize the variables
            total_timesteps = 0
            timesteps_since_eval = 0
            episode_num = 0
            done = True
            t0 = time.time()

            # Added by Sess10
            episode_reward = 0
            episode_timesteps = 0

            first_update = False

        # -------------------------------------------------------------------------------------
        # Sess10: TD3 TRAINING LOOP -- STARTS

        # ## Training
        # We start the main loop over 500,000 timesteps

        # Sess10: Removing the while loop and replacing with if condition if total_timesteps < max_timesteps:
        # while total_timesteps < max_timesteps:

        if total_timesteps < max_timesteps:
            # If the episode is done
            print("------TRAINING ITERATION START---------------------------------------")
            print("Current Time Steps value is: ", total_timesteps)
            #Train every 100 steps...basically allowing it to explore more
            if done or (total_timesteps>= 1000 and total_timesteps%train_n_steps == 0): # Sess10: Extra condition added total_timesteps> 1000 as else we will never reach done = True
            #Later, change it to 1000 or put some logic similar to done say every 1000 steps. Aim is to explore more
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    print("done block: Total Timesteps: {} Episode Num: {} Episode Reward: {}".format(total_timesteps, episode_num, episode_reward))
                    policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

                # We evaluate the episode and we save the policy
                if timesteps_since_eval >= eval_freq:
                    timesteps_since_eval %= eval_freq
                    # evaluations.append(evaluate_policy(self, env, policy))
                    #Sess10: Add a condition to reduce the freq of model saving
                    new_file_name = file_name + "_" + str(total_timesteps) #Added by Sess10
                    policy.save(new_file_name, directory="./pytorch_models")
                    # np.save("./results/%s" % (file_name), evaluations)

                # Modified by Sess10
                # When the training step is done, we reset the state of the environment
                # obs, _, _, _ = env.reset()  #Sess10: this is a 2D image (Black & White Image)
                #Note: In CarEnv, we dont need to reset the env, if done is true

                # Set the Done to False
                done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                _, CURRENT_1D_STATE_TO_BE_PASSED_TO_NN = getTransformedStateVector(self, currentState, obs,iS = input_size)

            # Before 10000 timesteps, we play random actions
            if total_timesteps < start_timesteps:
                # https://gym.openai.com/docs/ --> Spaces
                #The Discrete space allows a fixed range of non-negative numbers.
                # The Box space represents an n-dimensional box. HalfCheetah has Box(6,0) action_space
                # Sess 10: Car Steering wheel has one degree of freedom
                action = env.action_space.sample()
                # action = np.random.randint(0, number_actions)..>This was used in discrete DQN case

            else:  # After 10000 timesteps, we switch to the model

                # _, CURRENT_1D_STATE_TO_BE_PASSED_TO_NN = getTransformedStateVector(self, currentState, obs, iS = input_size)
                # _, obs = getTransformedStateVector(self, currentState, None)

                #Sess10: Implemented epsilon greedy algorithm to breakout from loop
                # Choosing an action to play
                if np.random.rand() < epsilon:
                    print("random action taken")
                    action = env.action_space.sample()  #Choosing a random action

                else:
                    # action = policy.select_action(np.array(obs))
                    action = policy.select_action(np.array(CURRENT_1D_STATE_TO_BE_PASSED_TO_NN))
                    # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                    if expl_noise != 0:
                        action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(
                            env.action_space.low, env.action_space.high)

                    print("policy action taken")

            # The agent performs the action in the environment, then reaches the next state and receives the reward
            new_obs, reward, done, _ = env.step(self, action)  # this 'new_obs' is a 2 D image (Black & White Image)

            if save_state_image:
                save_image(new_obs, total_timesteps) #Saving state images to a folder

            print("action taken: {} reward by step : {} done: {} ".format(action, reward, done))

            # Sess10 TBD:  This new state is a new frame....pass it some method to the get the modified state vector
            # Adding new game frame to the next state and deleting the oldest frame from next state

            nextState, NEXT_1D_STATE_TO_BE_PASSED_TO_NN = getTransformedStateVector(self, nextState, new_obs, iS = input_size)  # Returns 1D array of frame difference. Could be a neural network also
            #Note: Since, new_obs is going to be stored, it should be in. And ADD THE ORIENTATIONS ETC. AS WELL

            # We check if the episode is done
            done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

            # We increase the total reward
            episode_reward += reward

            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            # replay_buffer.add((obs, new_obs, action, reward, done_bool))
            replay_buffer.add((CURRENT_1D_STATE_TO_BE_PASSED_TO_NN, NEXT_1D_STATE_TO_BE_PASSED_TO_NN, action, reward, done_bool))

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            currentState = nextState

            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

            # # Lowering the epsilon
            if epsilon > minEpsilon:
                epsilon -= epsilonDecayRate

            print("------END------------------------------------------------------------")

        # Sess10:
        # total_timesteps < max_timesteps:
        if total_timesteps == max_timesteps: #Sess10: Save the model in the end, once all training iterations are over

            # We add the last policy evaluation to our list of evaluations and we save our model
            #Sess10:  Temporary disabled evaluations
            # evaluations.append(evaluate_policy(self, env, policy))
            if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
            # np.save("./results/%s" % (file_name), evaluations)

        #Sess10: TD3 TRAINING LOOP -- END

def evaluate_policy(self, env, policy, eval_episodes=10):
  avg_reward = 0.
  for _ in range(eval_episodes):
    obs = env.reset(self)
    done = False
    while not done:
      action = policy.select_action(np.array(obs))
      obs, reward, done, _ = env.step(action)
      avg_reward += reward
  avg_reward /= eval_episodes
  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("---------------------------------------")
  return avg_reward

#TD3 related method
# ## We create a new folder directory in which the final results (videos of the agent) will be populated
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Adding the painting tools
# def getTransformedStateVector(self, State_ND_Array, Observation, iS=(4,100,100)):
def getTransformedStateVector(self, State_ND_Array, Observation, iS):
    """

    Observation is the car_crop (2 D Black & White Image) of car_crop_width x car_crop_height
    State_ND_Array is a numpy array of number of stacked frames x car_crop_width x car_crop_height

    returns 1 D array of the state to be passed to TD3 Actor-Critic NN or to the Replay Buffer
    """

    # Adding new game frame to the next state and deleting the oldest frame from next state
    car_crop_width = iS[1] #100
    car_crop_height = iS[2] #100
    number_of_stacked_frame = iS[0] #4

    current_frame_number = number_of_stacked_frame - 1
    last_frame_number =  current_frame_number - 1

    # print("State_ND_Array: ", type(State_ND_Array) )
    # print("Observation: ", type(Observation) )
    # print("Observation shape is: ", Observation.shape )
    #
    # if Observation != None:  START
    state = np.reshape(Observation, (1, 1, car_crop_width, car_crop_height))
    State_ND_Array = np.append(State_ND_Array, state, axis=1)
    State_ND_Array = np.delete(State_ND_Array, 0, axis=1) #Now, this is a 4 d np array i.e. with 4 stacked frame
    # if Observation != None:  END

    # Gathering the state details for Training
    # current_frame = currentState[:, 3, :, :]
    # last_frame = currentState[:, 2, :, :]

    current_frame = State_ND_Array[:, current_frame_number, :, :]
    last_frame = State_ND_Array[:, last_frame_number, :, :]

    if (feature_extraction_mode == 'frame_diff'):

        diff_frame = current_frame - last_frame
        STATE_TO_BE_PASSED_TO_NN = diff_frame.flatten()  # IT IS A ONE DIMENSIONAL VECTOR

        # ADDITIONAL PARAMETER TO BE PASSED TO NN
        # goal_x, goal_y, orientation, distance_to_goal = env.get_goal(self)
        goal_x, goal_y, orientation, distance_to_goal,velocity, angle = env.get_goal(self)

        STATE_TO_BE_PASSED_TO_NN = np.append(STATE_TO_BE_PASSED_TO_NN, orientation)
        STATE_TO_BE_PASSED_TO_NN = np.append(STATE_TO_BE_PASSED_TO_NN, -orientation) # DIMENSION: 10002
        # STATE_TO_BE_PASSED_TO_NN = np.append(STATE_TO_BE_PASSED_TO_NN, distance_to_goal)  # DIMENSION: 10003

    elif (feature_extraction_mode == 'cnn_dcqn'):

        # feature_extrator_model_name = 'last_brain_54000.pth'
        feature_extraction_iS = (4, 100, 100)
        output_array = DCQN_FeatureExtractor.method_feature_extration(state=State_ND_Array, iS=feature_extraction_iS, nb_action=3, model_name=feature_extrator_model_name)
        STATE_TO_BE_PASSED_TO_NN = output_array.flatten()
        # (40,)
        # ADDITIONAL PARAMETER TO BE PASSED TO NN
        # goal_x, goal_y, orientation, distance_to_goal = env.get_goal(self)
        goal_x, goal_y, orientation, distance_to_goal, velocity, angle = env.get_goal(self)

        STATE_TO_BE_PASSED_TO_NN = np.append(STATE_TO_BE_PASSED_TO_NN, orientation)
        STATE_TO_BE_PASSED_TO_NN = np.append(STATE_TO_BE_PASSED_TO_NN, -orientation) # DIMENSION: 42
        # STATE_TO_BE_PASSED_TO_NN = np.append(STATE_TO_BE_PASSED_TO_NN, distance_to_goal)  # DIMENSION: 43
        STATE_TO_BE_PASSED_TO_NN = np.append(STATE_TO_BE_PASSED_TO_NN, velocity) # DIMENSION: 44
        STATE_TO_BE_PASSED_TO_NN = np.append(STATE_TO_BE_PASSED_TO_NN, angle)# DIMENSION: 45

    elif (feature_extraction_mode == 'cnn_mnist'):
        # Pass the Current Frame and Last Frame and take the difference OR
        # Pass all the 4 frames one by one and stack them --- flatten them
        # Or should we try with Single Frame and along with Orientation

        output_array = MNIST_FeatureExtractor.method_feature_extration(state=State_ND_Array, iS=iS, model_name = feature_extrator_model_name)
        STATE_TO_BE_PASSED_TO_NN = output_array.flatten()  # 4*50 = 200

        print("Sess10  STATE_TO_BE_PASSED_TO_NN before append", STATE_TO_BE_PASSED_TO_NN.shape)

        # ADDITIONAL PARAMETER TO BE PASSED TO NN
        # goal_x, goal_y, orientation, distance_to_goal = env.get_goal(self)
        goal_x, goal_y, orientation, distance_to_goal, velocity, angle = env.get_goal(self)

        STATE_TO_BE_PASSED_TO_NN = np.append(STATE_TO_BE_PASSED_TO_NN, orientation)
        STATE_TO_BE_PASSED_TO_NN = np.append(STATE_TO_BE_PASSED_TO_NN, -orientation) # DIMENSION: 202

        print("Sess10  STATE_TO_BE_PASSED_TO_NN after orientation", STATE_TO_BE_PASSED_TO_NN.shape)

        # STATE_TO_BE_PASSED_TO_NN = np.append(STATE_TO_BE_PASSED_TO_NN, distance_to_goal)  # DIMENSION: 43
        STATE_TO_BE_PASSED_TO_NN = np.append(STATE_TO_BE_PASSED_TO_NN, velocity) # DIMENSION: 203

        print("Sess10  STATE_TO_BE_PASSED_TO_NN after velocity", STATE_TO_BE_PASSED_TO_NN.shape)

        print("Sess10  velocity is ", velocity)

        STATE_TO_BE_PASSED_TO_NN = np.append(STATE_TO_BE_PASSED_TO_NN, angle)# DIMENSION: 204

        print("Sess10  angle is ", angle)
        print("Sess10  STATE_TO_BE_PASSED_TO_NN", STATE_TO_BE_PASSED_TO_NN.shape)

    return State_ND_Array, STATE_TO_BE_PASSED_TO_NN

# Making a function that will initialize game states
def resetPixelStates(self):

    rows = input_size[1] # 100
    columns = input_size[2] # 100
    nLastStates = input_size[0] #4

    # currentState = np.zeros((1, env.nRows, env.nColumns, nLastStates))
    # currentState = np.zeros((1, rows, columns, nLastStates))
    currentState = np.zeros((1, nLastStates, rows, columns))

    for i in range(nLastStates):

        # last_signal, last_reward, episode_over, info = env.reset(self)
        pixel_state = env.get_state(self)  #This is a 2D image (Black & White)

        # currentState[:, :, :, i] = pixel_state  # a rows x columns numpy array
        currentState[:, i, :, :] = pixel_state  # a rows x columns numpy array

    return currentState, currentState


class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):


        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.train, 1.0/60.0)  #Note that First method getting called is 'Game.update' (now called Game.train())

        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear', size = (50,20))
        savebtn = Button(text = 'save', pos = (parent.width, 0), size = (50,20))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0), size = (50,20))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        pass
        # global sand
        # global longueur, largeur
        # self.painter.canvas.clear()
        # sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        # brain.save()
        # brain_dcqn.save()
        # plt.plot(scores)
        # plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        # brain.load()
        # brain_dcqn.load()


#Sess10's step method



# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
