import turtle
import random
import math

class Shapes(turtle.Turtle):
    """
    For testing use:
    screen = turtle.Screen()
    screen.setup(width=500, height=500)
    s = Shapes()
    shapes = [
        {'shape': 'triangle', 'color': 'red', 'size': 'big'},
        {'shape': 'circle', 'color': 'blue', 'size': 'big'}
    ]
    s.draw_multiple(shapes)
    turtle.done()

    """
    
    def __init__(self, speed=0):
        super().__init__()
        self.speed(speed)
        self.hideturtle()

        self.object = {
            'circle': self.my_circle,
            'square': self.square,
            'triangle': self.triangle,
            'diamond': self.diamond,
            'hexagon': self.hexagon,
            'octagon': self.octagon
        }

        # Consistent sizes
        self.size_map = {
            'small': 60,
            'medium': 90,
            'big': 120
        }

    def move(self, go_to):
        self.penup()
        self.goto(go_to)
        self.pendown()
        self.setheading(0)
        return self

    # ---------------- SHAPES ---------------- #

    def square(self, go_to, colour, size):
        self.color(colour, colour)
        self.move((go_to[0] - size / 2, go_to[1] - size / 2))

        self.begin_fill()
        for _ in range(4):
            self.forward(size)
            self.left(90)
        self.end_fill()

    def my_circle(self, go_to, colour, size):
        self.color(colour, colour)
        self.move((go_to[0], go_to[1] - size / 2))

        self.begin_fill()
        self.circle(size / 2)
        self.end_fill()

    def triangle(self, go_to, colour, size):
        self.color(colour, colour)

        height = (math.sqrt(3) / 2) * size
        self.move((go_to[0] - size / 2, go_to[1] - height / 2))

        self.begin_fill()
        for _ in range(3):
            self.forward(size)
            self.left(120)
        self.end_fill()

    def diamond(self, go_to, colour, size):
        self.color(colour, colour)

        self.move(go_to)
        self.setheading(45)

        self.begin_fill()
        for _ in range(4):
            self.forward(size)
            self.left(90)
        self.end_fill()

        self.setheading(0)

    def hexagon(self, go_to, colour, size):
        self.color(colour, colour)

        side = size / 2
        self.move((go_to[0] - side, go_to[1]))

        self.begin_fill()
        for _ in range(6):
            self.forward(side)
            self.left(60)
        self.end_fill()

    def octagon(self, go_to, colour, size):
        self.color(colour, colour)

        side = size / 3
        self.move((go_to[0] - side, go_to[1]))

        self.begin_fill()
        for _ in range(8):
            self.forward(side)
            self.left(45)
        self.end_fill()

    # ---------------- GRID SYSTEM ---------------- #

    def generate_grid_positions(self, screen_size, count):
        margin = 80
        usable = screen_size - (2 * margin)

        cols = int(math.ceil(math.sqrt(count)))
        rows = int(math.ceil(count / cols))

        cell_w = usable / cols
        cell_h = usable / rows

        positions = []

        for r in range(rows):
            for c in range(cols):
                x = -screen_size / 2 + margin + (c + 0.5) * cell_w
                y = -screen_size / 2 + margin + (r + 0.5) * cell_h
                positions.append((x, y))

        random.shuffle(positions)
        return positions[:count]

    # ---------------- DRAW MULTIPLE ---------------- #

    def draw_multiple(self, shapes_list, screen_size=500):
        """
        shapes_list = [
            {'shape': 'triangle', 'color': 'red', 'size': 'medium'},
            {'shape': 'circle', 'color': 'blue', 'size': 'small'},
            ...
        ]
        """

        positions = self.generate_grid_positions(screen_size, len(shapes_list))

        for i in range(len(shapes_list)):
            shape_data = shapes_list[i]

            shape_name = shape_data['shape']
            color = shape_data['color']
            size = self.size_map[shape_data['size']]

            pos = positions[i]

            self.object[shape_name](pos, color, size)