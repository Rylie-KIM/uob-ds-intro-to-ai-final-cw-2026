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
    
    ############
    ############
    # The shapes ######
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

    def get_dims(self, shape:str, size:int) -> tuple[int, int]:
        if shape =='square' or shape =='circle':
            return size, size
        elif shape == 'triangle':
            s = (size*3)/2
            A = math.sqrt((s * (s - size)**3))
            height =  A / ((1/2)*size)
            width = size
            return width, height
        elif shape =='diamond':
            d = ((size**2) + (size**2))
            width = math.sqrt(d)
            height = width
            return width, height
        elif shape =='hexagon':
            # Overlapping issue for either hexagon or octagon
            width = 2 * size
            a = (math.sqrt(3)/2) * size
            height = 2*a
            return width, height
        elif shape == 'octagon':
            # Overlapping issue for either hexagon or octagon
            width = size * (math.sqrt((4 + 2*(math.sqrt(2)))))
            height = width
            return width, height

    def coordinates(self,x:int, y:int, relation:str, s1:int, s2:int,o1:str, o2:str, screen_size:Tuple[int,int]) -> tuple[int,int]:
        """
        The coordinates method begins by calculating the available space ie how much space each of the shapes takes up, defined by the size parameter
        This is done in order to make the x or y coordinates generate relative to the size of each shape individually as to prevent overlapping.

        """
        

        screen_x_min, screen_x_max, screen_y_min,screen_y_max = -screen_size[0]//2, screen_size[0]//2, -screen_size[1]//2, screen_size[1]//2
        width1, height1 = self.get_dims(o1, s1)
        width2, height2 = self.get_dims(o2, s2)
        # Added 5 padding between each shape to ensure gap
        x_space = int((width1/2) + (width2/2)) + 10
        # Should height2 be divided by 2?
        y_space = int((height1/2) + (height2/2)) + 10
        valid_range_x = 0
        
        if relation == 'above':
            # added 10 for padding
            valid_range_y = int((y - (screen_y_min + (height2/2)))-10)
            # i shoudl definitely add a screen width param to generate larger variance in shape distances
            if y_space > valid_range_y:
                return None
            x2, y2 = x + random.randint(-30,30), y - random.randint(y_space, valid_range_y)
            return x2,y2
        elif relation =='below':
            valid_range_y = int((screen_y_max - (y + height2/2)) - 10)
            if y_space > valid_range_y:
                return None
            x2, y2 = x + random.randint(-30,30), y + random.randint(y_space,valid_range_y)
            return x2, y2
        elif relation == 'right of':
            valid_range_x = int((x - (screen_x_min + (width2/2)) - (width1/2))-10)
            if x_space > valid_range_x:
                return None
            x2, y2 = x - random.randint(x_space,valid_range_x), y + random.randint(-30,30)
            return x2, y2
        elif relation == 'left of':
            valid_range_x = int((screen_x_max - (x + height2/2) -10))
            if x_space > valid_range_x:
                return None
            x2, y2 = x + random.randint(x_space,valid_range_x), y + random.randint(-30,30)
            return x2,y2

        
    def draw(self,s1:int, c1:str, o1:str, s2:int, c2:str, o2:str, rel:str, screen_size:Tuple[int,int]):
        """
        The .draw method takes a mix of str and int inputs to corresponding to size, colour and object_type for one of two objects
        x, y coordinates are initialised randomly for the first shape, ie the first shape is placed randomly on the page
        The coodinates of the second shape are then generated randomly, but within a specific range to satisfy the relation parameter.
        ie where relation is 'above' the second shapes coordinates are generated in the area bellow the shape.
        
        """
        x_min, x_max, y_min,y_max = -screen_size[0]//2, screen_size[0]//2, -screen_size[1]//2, screen_size[1]//2
        # Converts size from str ie 'big' to int value
        s1 = self.size[s1]
        s2 = self.size[s2]
        self.hideturtle()
        o1_width, o1_height = self.get_dims(o1, s1)
        while True:
            go_to = random.randint(int(x_min + o1_width), int(x_max - o1_width)), random.randint(int(y_min + o1_height), int(y_max - o1_height))
            shape = self.object[o1]
            shape(go_to, c1, s1)
            
            go_to2 = self.coordinates(go_to[0],go_to[1],rel, s1, s2, o1, o2,screen_size)
            if go_to2 == None:
                self.screen.clear()
                continue
            shape2 = self.object[o2]
            shape2(go_to2, c2, s2)
            break
        return self

SCREEN_SIZE = (500, 500)
screen = turtle.Screen()
screen.setup(width=SCREEN_SIZE[0], height=SCREEN_SIZE[1])
s = Shapes()
# s.hexagon((50,75), 'orange', 'big')
s.draw(s1='small', c1='red', o1='octagon', s2='small', c2='black', o2='hexagon', rel='left of', screen_size=SCREEN_SIZE)
turtle.done()