import turtle

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
    
    def draw(self, object, go_to, colour):
        self.hideturtle()
        shape = self.object[object]
        shape(go_to, colour)
        return self


screen = turtle.Screen()
screen.setup(width=500, height=500)
s = Shapes()
s.draw('square', (100,100), 'red').draw('triangle', (-80, -50), 'yellow')
turtle.done()
