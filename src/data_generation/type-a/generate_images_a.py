import turtle
import random

class Shapes(turtle.Turtle):
    """ 
    Pixel range is determined by screen = turtle.Screen() , screen.setup(width=500, height=500)
    Width is ± width/2
    Height is ± height/2
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

    def square(self, go_to:tuple[int,int], colour:str, size:int):
        """
        square((50,75), 'red', 100)
        """
        
        size = self.size[size]
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
        size = self.size[size]
        go_to = go_to[0], go_to[1] - (size/2)
        self.color(colour, colour)
        self.move(go_to)
        self.begin_fill()
        self.circle(size/2)
        self.end_fill()
        return self
    
    def triangle(self, go_to:tuple[int,int], colour:str, size:int):
        """
        triangle((50,75), 'red', 100)
        """

        size = self.size[size]
        go_to = go_to[0] - (size/2), go_to[1] - (size/2)
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
        diagmond((50,75), 'black', 120))
        
        """
        size = self.size[size]
        go_to = go_to[0] - (size/2), go_to[1] - (size/2)
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
        size = self.size[size]
        go_to = go_to[0] - (size/2), go_to[1] - (size/2)
        self.color(colour, colour)
        self.move(go_to)
        self.begin_fill()
        for i in range(6):
            self.forward(size/2)
            self.left(360/6)
        self.end_fill()
        return self
    
    def octagon(self, go_to:tuple[int,int], colour:str, size:int):
        """
        octagon((50,75), 'blue', 50))
        """
        size = self.size[size] // 2.5
        go_to = go_to[0] - (size/2), go_to[1] - (size/2)
        self.color(colour, colour)
        self.move(go_to)
        self.begin_fill()
        for i in range(8):
            self.forward(size)
            self.left(45)
        self.end_fill()
        return self
        
    def draw1(self, object, colour, size):
        go_to = random.randint(-250, 250), random.randint(-250, 250)
        self.hideturtle()
        shape = self.object[object]
        shape(go_to, colour, size)
        return self
    
    def draw(self,s1:int, c1:str, o1:str, s2:int, c2:str, o2:str, rel:str):
        """
        The .draw method takes a mix of str and int inputs to corresponding to size, colour and object_type for one of two objects
        x, y coordinates are initialised randomly for the first shape, ie the first shape is placed randomly on the page
        The coodinates of the second shape are then generated randomly, but within a specific range to satisfy the relation parameter.
        ie where relation is 'above' the second shapes coordinates are generated in the area bellow the shape.
        
        """
        self.hideturtle()
        x, y = random.randint(-180, 180), random.randint(-180, 180)
        shape = self.object[o1]
        shape((x,y), c1, s1)
        x2, y2 = self.coordinates(x,y,rel, s1, s2)
        shape2 = self.object[o2]
        shape2((x2,y2), c2, s2)
        return self
    
    def coordinates(self,x:int, y:int, relation:str, s1:int, s2:int):
        """
        The coordinates method begins by calculating the available space ie how much space each of the shapes takes up, defined by the size parameter
        This is done in order to make the x or y coordinates generate relative to the size of each shape individually as to prevent overlapping.

        """
        space = (self.size[s1]//2) + (self.size[s2]//2)
        if relation == 'above':
            x2, y2 = x + random.randint(-30,30), y - random.randint(space,space + 100)
            return x2,y2
        elif relation =='below':
            x2, y2 = x + random.randint(-30,30), y + random.randint(space,space + 100)
            return x2, y2
        elif relation == 'right of':
            x2, y2 = x - random.randint(space,space + 100), y + random.randint(-30,30)
            return x2, y2
        elif relation == 'left of':
            x2, y2 = x + random.randint(space,space + 100), y + random.randint(-30,30)
            return x2,y2

