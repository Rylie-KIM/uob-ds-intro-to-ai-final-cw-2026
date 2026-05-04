import turtle
import random
import math
from typing import Tuple, List

class Shapes(turtle.Turtle):
    """ 
    Pixel range is determined by screen = turtle.Screen() , screen.setup(width=500, height=500)
    Width is ± width/2
    Height is ± height/2

    For testing use: 

            screen = turtle.Screen()
            screen.setup(width=500, height=500)
            s = Shapes()
            # s.hexagon((50,75), 'orange', 'big')
            s.draw(s1='small', c1='red', o1='octagon', s2='small', c2='black', o2='hexagon', rel='above')
            turtle.done()

    """
    def __init__(self, speed=0):
        super().__init__()
        self.speed(speed)
        self.screen = turtle.Screen()
        self.object = {'circle': self.my_circle,
                      'square':self.square,
                      'triangle':self.triangle,
                      'diamond':self.diamond,
                      'hexagon':self.hexagon,
                      'octagon':self.octagon}
        # This initialises the sizes on start so would need to create a new instance of the class each time for unique sizes.
        self.size = {'big':random.randint(130,150),
                     'medium':random.randint(80,100),
                     'small':random.randint(30,50)}

    def move(self, go_to:tuple[int,int]):
        self.penup()
        self.goto(go_to)
        self.pendown()
        return self

    def square(self, go_to:tuple[int,int], colour:str, size:int):
        """
        square((50,75), 'red', 100)
        """
        # This makes the initial go_to point the center.
        go_to = go_to[0] - (size/2), go_to[1] - (size/2)
        self.color(colour, colour)
        self.move(go_to)
        self.begin_fill()
        for i in range(4):
            self.forward(size)
            self.left(90)
        self.end_fill()
        return self

    def my_circle(self, go_to:tuple[int,int], colour:str, size:int):
        """
        my_circle((50,75), 'red', 100)
        """
        go_to = go_to[0] - (size/2), go_to[1] - (size/2)
        self.color(colour, colour)
        self.move(go_to)
        self.begin_fill()
        self.circle(size/2)
        self.end_fill()
        return self

    def triangle(self, go_to:tuple[int,int], colour:str, size:int):
        """
        triangle((50,75), 'red', 100)
        Have to use Herons formula to calculate the height of a triangle area = (1/2)(base)(height)
        rearranges to..... 
        """
        width, height = self.get_dims('triangle', size)
        go_to = go_to[0] - (width/2), go_to[1] - (height/2)
        self.color(colour, colour)
        self.move(go_to)
        self.begin_fill()
        for i in range(3):
            self.forward(size)
            self.left(120)
        self.end_fill()
        return self
    
    def diamond(self, go_to:tuple[int,int], colour:str, size:int):
        """ 
        diamond((50,75), 'black', 120))
        """
        width, height = self.get_dims('diamond', size)
        go_to = go_to[0] - (width/2), go_to[1] - (height/2)
        self.color(colour, colour)
        self.move(go_to)
        self.begin_fill()
        self.left(45)
        for i in range(4):
            self.forward(size)
            self.left(90)
        self.end_fill()
        return self
    
    def hexagon(self, go_to:tuple[int,int], colour:str, size:int):
        """
        hexagon((50,75), 'orange', 60))
        """
        width, height = self.get_dims('hexagon', size)
        go_to = go_to[0] - (width/2), go_to[1] - (height/2)
        self.color(colour, colour)
        self.move(go_to)
        self.begin_fill()
        for i in range(6):
            self.forward(size)
            self.left(360/6)
        self.end_fill()
        return self
    
    def octagon(self, go_to:tuple[int,int], colour:str, size:int):
        """
        octagon((50,75), 'blue', 50))
        """
        width, height = self.get_dims('octagon', size)
        go_to = go_to[0] - (width/2), go_to[1] - (height/2)
        self.color(colour, colour)
        self.move(go_to)
        self.begin_fill()
        for i in range(8):
            self.forward(size)
            self.left(45)
        self.end_fill()
        return self