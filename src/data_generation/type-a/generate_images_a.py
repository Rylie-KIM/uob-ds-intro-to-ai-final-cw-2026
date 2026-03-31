import turtle
import random
import math

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
            width = 2 * size
            a = (math.sqrt(3)/2) * size
            height = 2*a
            return width, height
        elif shape == 'octagon':
            width = size * (math.sqrt((4 + 2*(math.sqrt(2)))))
            height = width
            return width, height

    def coordinates(self,x:int, y:int, relation:str, s1:int, s2:int,o1:str, o2:str) -> tuple[int,int]:
        """
        The coordinates method begins by calculating the available space ie how much space each of the shapes takes up, defined by the size parameter
        This is done in order to make the x or y coordinates generate relative to the size of each shape individually as to prevent overlapping.

        """
        width1, height1 = self.get_dims(o1, s1)
        width2, height2 = self.get_dims(o2, s2)
        x_space = int((width1/2) + (width2/2))
        y_space = int((height1/2) + (height2))
        if relation == 'above':
            # i shoudl definitely add a screen width param to generate larger variance in shape distances
            x2, y2 = x + random.randint(-30,30), y - random.randint(y_space,y_space + 100)
            return x2,y2
        elif relation =='below':
            x2, y2 = x + random.randint(-30,30), y + random.randint(y_space,y_space + 100)
            return x2, y2
        elif relation == 'right of':
            x2, y2 = x - random.randint(x_space,x_space + 100), y + random.randint(-30,30)
            return x2, y2
        elif relation == 'left of':
            x2, y2 = x + random.randint(x_space,x_space + 100), y + random.randint(-30,30)
            return x2,y2
        
    def draw(self,s1:int, c1:str, o1:str, s2:int, c2:str, o2:str, rel:str):
        """
        The .draw method takes a mix of str and int inputs to corresponding to size, colour and object_type for one of two objects
        x, y coordinates are initialised randomly for the first shape, ie the first shape is placed randomly on the page
        The coodinates of the second shape are then generated randomly, but within a specific range to satisfy the relation parameter.
        ie where relation is 'above' the second shapes coordinates are generated in the area bellow the shape.
        
        """
        # Converts size from str ie 'big' to int value
        s1 = self.size[s1]
        s2 = self.size[s2]
        self.hideturtle()
        # Randomly chose 180, may want to set range to screen width - shape width/length
        go_to = random.randint(-180, 180), random.randint(-180, 180)
        shape = self.object[o1]
        shape(go_to, c1, s1)
        go_to2 = self.coordinates(go_to[0],go_to[1],rel, s1, s2, o1, o2)
        shape2 = self.object[o2]
        shape2(go_to2, c2, s2)
        return self
