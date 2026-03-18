import turtle

class Shapes(turtle.Turtle):
    """ 
    Pixel range is determined by screen = turtle.Screen() , screen.setup(width=500, height=500)
    Width is ± width/2
    Height is ± height/2
    """
    def __init__(self, speed=0):
        super().__init__()
        self.speed(speed)

    def move(self, go_to):
        self.penup()
        self.goto(go_to)
        self.pendown()
        return self

    def square(self, go_to):
        self.move(go_to)
        for i in range(4):
            self.forward(100)
            self.left(90)
        return self

    def my_circle(self, go_to):
        """ 
        t.circle allows you to specify circle radius thus could add size feature as parameter
        """
        self.move(go_to)
        self.circle(50)
        return self
    
    def triangle(self, go_to):
        self.move(go_to)
        for i in range(3):
            self.forward(100)
            self.left(120)
        return self

screen = turtle.Screen()
screen.setup(width=500, height=500)
s = Shapes()
s.triangle((50,100)).square((30,20)).my_circle((150, 150))
turtle.done()
