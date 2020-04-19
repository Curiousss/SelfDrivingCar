'''

    # Self Driving Car by Meera - Implementing TD3 with the custom Car Env
    @author = "Meera"
    @copyright = "Copyright 2020, The Self Driving Car Project RL Custom Environment with TD3"
    @license = "GPL"
    @version = "1.0.1"
    @maintainer = "Sess102020"
'''

# Importing the libraries
import gym
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
from random import randint

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

# Importing the Dqn object from our AI in ai.py
# from ai import Dqn


# Creating the car class. This projects the Car Agent in the CarEnv
class Car(Widget):
    '''

        # Self Driving Car by Meers - Implementing TD3 with the custom Car Env
        @author = "Meera"
        @copyright = "Copyright 2020, The Self Driving Car Project RL Custom Environment with TD3"
        @license = "GPL"
        @version = "1.0.1"
        @maintainer = "Sess102020"
    '''

    # angle = NumericProperty(0)
    # rotation = NumericProperty(0)

    angle = NumericProperty(0.0)
    rotation = NumericProperty(0.0)


    velocity_x = NumericProperty(0.0)
    velocity_y = NumericProperty(0.0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    sensor1_x = NumericProperty(0.0)
    sensor1_y = NumericProperty(0.0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)

    sensor2_x = NumericProperty(0.0)
    sensor2_y = NumericProperty(0.0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)

    sensor3_x = NumericProperty(0.0)
    sensor3_y = NumericProperty(0.0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)

    #Sess10: These are the Signals
    signal1 = NumericProperty(0.0)
    signal2 = NumericProperty(0.0)
    signal3 = NumericProperty(0.0)

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass
class Goal1(Widget):
    pass
class Goal2(Widget):
    pass
class Goal3(Widget):
    pass