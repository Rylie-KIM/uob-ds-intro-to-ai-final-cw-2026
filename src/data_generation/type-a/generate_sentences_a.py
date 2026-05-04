import turtle
from generate_images_a import Shapes
import os
import csv
from pathlib import Path

screen = turtle.Screen()
screen.setup(width=700, height=700)
objects = ['circle','triangle','square','diamond','hexagon', 'octagon']
colours = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black']
size = ['big', 'medium', 'small']
relations = ['above', 'below', 'left of', 'right of']
sentence_templates = [
    'a {size1} {colour1} {object1} is {relation} a {size2} {colour2} {object2}',
    'the {size2} {colour2} {object2} is positioned {relation} a {size2} {colour2} {object2}',
    'a {size2} {colour2} {object2} can be seen {relation} a {size2} {colour2} {object2}',
    '{relation} a {size2} {colour2} {object2} is a {size1} {colour1} {object1}',
]

def generator(objects:list, colours:list, size:list, sentence_templates:list) -> list:
    """
    Creates list of structured sentences where each bundle represents a concatenated str of <size> <colour> <object> 
    Output is list of structured sentences
    """
    i = 0
    for template in sentence_templates:
        for o1 in objects:
            for c1 in colours:
                for s1 in size:
                    for o2 in objects:
                        for c2 in colours:
                            for s2 in size:
                                for rel in relations:
                                    if f'{s1} {c1} {o1}' == f'{s2} {c2} {o2}':
                                        continue
                                    sentence = template.format(size1=s1, colour1=c1,object1=o1,relation=rel,size2=s2,colour2=c2,object2=o2)
                                    s = Shapes()
                                    s.getscreen().clearscreen()
                                    s.draw(s1=s1, c1=c1, o1=o1, s2=s2, c2=c2, o2=o2,rel=rel)

    return 'done'
