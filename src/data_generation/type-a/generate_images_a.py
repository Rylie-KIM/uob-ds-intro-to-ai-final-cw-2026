import turtle
import random
class Shapes(turtle.Turtle):
    """ 
    Pixel range is determined by screen = turtle.Screen() , screen.setup(width=500, height=500)
    Width is ± width/2
    Height is ± height/2

    Coordinate ranges for each segment in 500x500 shape


            ([-250:-83.3], [250:83.3])    | ([-83.3:83.3], [250:83.3])   |  ([83.3:250], [250:83.3]) 
        ----------------------------------------------------------------------------------------------------
     
            ([-250:-83.3], [83.3:-83.3]   | ([-83.3:83.3], [-83.3:83.3])   |  ([83.3:250], [-83.3:83.3]) 
        ----------------------------------------------------------------------------------------------------
       
            ([-250:-83.3], [-83.3:-250])   | ([-83.3:83.3], [-250:-83.3])   | ([83.3:250], [-250:-83.3]) 
    """
    def __init__(self, speed=0):
        super().__init__()
        self.speed(speed)
        self.object = {'circle': self.my_circle,
                      'square':self.square,
                      'triangle':self.triangle,
                      'diamond':self.diamond,
                      'hexagon':self.hexagon,
                      'octogon':self.octogon}
        # This initialises the sizes on start so would need to create a new instance of the class each time for unique sizes.
        self.size = {'big':random.randint(130,150),
                     'medium':random.randint(80,100),
                     'small':random.randint(30,50)}

    def move(self, go_to):
        self.penup()
        self.goto(go_to)
        self.pendown()
        return self

    def square(self, go_to, colour, size):
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

    def my_circle(self, go_to, colour, size):
        """ 
        self.circle allows you to specify circle radius thus could add size feature as parameter
        """
        size = self.size[size]
        go_to = go_to[0], go_to[1] - (size/2)
        self.color(colour, colour)
        self.move(go_to)
        self.begin_fill()
        self.circle(size/2)
        self.end_fill()
        return self
    
    def triangle(self, go_to, colour, size):
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
    
    def diamond(self, go_to, colour, size):
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
    
    def hexagon(self, go_to, colour, size):
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
    
    def octogon(self, go_to, colour, size):
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
    
    def draw(self,s1, c1, o1, s2, c2, o2, rel):
        self.hideturtle()
        x, y = random.randint(-180, 180), random.randint(-180, 180)
        shape = self.object[o1]
        shape((x,y), c1, s1)
        x2, y2 = self.coordinates(x,y,rel, s1, s2)
        shape2 = self.object[o2]
        shape2((x2,y2), c2, s2)
        return self
    
    def coordinates(self,x, y, relation, s1, s2):
        space = (self.size[s1]//2) + (self.size[s2]//2)
        if relation == 'above':
            x2, y2 = x + random.randint(-30,30), y - random.randint(space,space + 100)
            return x2,y2
        elif relation =='below':
            x2, y2 = x + random.randint(-30,30), y + random.randint(space,space + 100)
            return x2, y2
        elif relation == 'right of':
            x2, y2 = x + random.randint(space,space + 100), y + random.randint(-30,30)
            return x2, y2
        elif relation == 'left of':
            x2, y2 = x - random.randint(space,space + 100), y + random.randint(-30,30)
            return x2,y2

screen = turtle.Screen()
screen.setup(width=700, height=700)
s = Shapes()
s.draw(s1='big', c1='blue', o1='octogon', s2='big', c2='orange', o2='circle',rel='left of')
# s.draw1('octogon', 'red', 'big')
turtle.done()