import turtle
import math

class ShapesGenerator(turtle.Turtle):

    """
    For testing use: 
    
    screen = turtle.Screen()
    screen.setup(width=500, height=500)
    s = ShapesGenerator()
    s.draw_relation(
        'big', 'red', 'diamond',
        'medium', 'blue', 'hexagon',
        'above'
    )
    turtle.done()

    """

    def __init__(self):
        super().__init__()
        self.hideturtle()
        self.speed(0)

        # mapping shape names to functions
        self.object = {
            'circle': self.my_circle,
            'square': self.square,
            'triangle': self.triangle,
            'diamond': self.diamond,
            'hexagon': self.hexagon,
            'octagon': self.octagon
        }

        # fixed sizes so they stay consistent
        self.size_map = {
            'small': 40,
            'medium': 70,
            'big': 100
        }

    def move(self, pos):
        self.penup()
        self.goto(pos)
        self.pendown()
        self.setheading(0)


    # ---------- shapes ----------

    def square(self, pos, color, size):
        self.color(color)
        self.move((pos[0] - size/2, pos[1] - size/2))

        self.begin_fill()
        for i in range(4):
            self.forward(size)
            self.left(90)
        self.end_fill()

    def my_circle(self, pos, color, size):
        self.color(color)
        self.move((pos[0], pos[1] - size/2))

        self.begin_fill()
        self.circle(size/2)
        self.end_fill()

    def triangle(self, pos, color, size):
        self.color(color)

        h = (math.sqrt(3)/2) * size
        self.move((pos[0] - size/2, pos[1] - h/2))

        self.begin_fill()
        for i in range(3):
            self.forward(size)
            self.left(120)
        self.end_fill()

    def diamond(self, pos, color, size):
        self.color(color)

        self.move(pos)
        self.setheading(45)

        self.begin_fill()
        for i in range(4):
            self.forward(size / 1.5)
            self.left(90)
        self.end_fill()

        self.setheading(0)

    def hexagon(self, pos, color, size):
        self.color(color)

        side = size / 2
        self.move((pos[0] - side, pos[1]))

        self.begin_fill()
        for i in range(6):
            self.forward(side)
            self.left(60)
        self.end_fill()

    def octagon(self, pos, color, size):
        self.color(color)

        side = size / 3
        self.move((pos[0] - side, pos[1]))

        self.begin_fill()
        for i in range(8):
            self.forward(side)
            self.left(45)
        self.end_fill()


    # ---------- positioning ----------

    def get_positions(self, relation):
        offset = 120  # just a value to separate shapes

        if relation == "above":
            return (0, offset), (0, -offset)
        elif relation == "below":
            return (0, -offset), (0, offset)
        elif relation == "left of":
            return (-offset, 0), (offset, 0)
        elif relation == "right of":
            return (offset, 0), (-offset, 0)


    # ---------- main function ----------

    def draw_relation(self, s1, c1, o1, s2, c2, o2, relation):

        pos1, pos2 = self.get_positions(relation)

        size1 = self.size_map[s1]
        size2 = self.size_map[s2]

        # draw both shapes
        self.object[o1](pos1, c1, size1)
        self.object[o2](pos2, c2, size2)