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
                      'triangle':self.triangle}
        self.size = {'big':100,
                     'medium':70,
                     'small':50}
        """
        self.relations = {
            'above':self.draw_above,
            'below':self.draw_bellow,
            'left of':self.draw_left,
            'right of': self.draw_right
        }
        """

    def move(self, go_to):
        self.penup()
        self.goto(go_to)
        self.pendown()
        return self

    def square(self, go_to, colour):
        self.color(colour, colour)
        self.begin_fill()
        self.move(go_to)
        for i in range(4):
            self.forward(100)
            self.left(90)
        self.end_fill()
        return self

    def my_circle(self, go_to, colour):
        """ 
        self.circle allows you to specify circle radius thus could add size feature as parameter
        """
        self.color(colour, colour)
        self.begin_fill()
        self.move(go_to)
        self.circle(50)
        self.end_fill()
        return self
    
    def triangle(self, go_to, colour):
        self.color(colour, colour)
        self.begin_fill()
        self.move(go_to)
        for i in range(3):
            self.forward(100)
            self.left(120)
        self.end_fill()
        return self
    
    def draw1(self, object, colour):

        go_to = random.randint(-250, 250), random.randint(-250, 250)
        self.hideturtle()
        shape = self.object[object]
        shape(go_to, colour)
        return self
    
    def draw(self,s1, c1, o1, s2, c2, o2, rel):
        self.hideturtle()
        # relation = self.relations[rel]
        x, y = random.randint(-180, 180), random.randint(-180, 180)
        shape = self.object[o1]
        shape((x,y), c1)
        x2, y2 = self.coordinates(x,y,rel)
        shape2 = self.object[o2]
        shape2((x2,y2), c2)
        return self
    
    def coordinates(self,x, y, relation):
        if relation == 'above':
            x2, y2 = x + random.randint(-30,30), y - random.randint(50,150)
            return x2,y2
        elif relation =='below':
            x2, y2 = x + random.randint(-30,30), y + random.randint(50,150)
            return x2, y2
        elif relation == 'right of':
            x2, y2 = x + random.randint(50,150), y + random.randint(-30,30)
            return x2, y2
        elif relation == 'left of':
            x2, y2 = x - random.randint(50,150), y + random.randint(-30,30)
            return x2,y2

screen = turtle.Screen()
screen.setup(width=500, height=500)
s = Shapes()
s.draw(s1='big', c1='red', o1='circle', s2='small', c2='blue', o2='square',rel='above')
turtle.done()


# hello workld