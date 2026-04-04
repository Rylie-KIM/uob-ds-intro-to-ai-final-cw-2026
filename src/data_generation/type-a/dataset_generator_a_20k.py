import csv
import random
import turtle
from itertools import product
from pathlib import Path

from relation_shapes_generator_a import ShapesGenerator

NUM_OBSERVATIONS = 20000
RANDOM_SEED = 42
SCREEN_SIZE = (700, 700)

objects = ['circle', 'triangle', 'square', 'diamond', 'hexagon', 'octagon']
colours = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black']
sizes = ['big', 'medium', 'small']
relations = ['above', 'below', 'left of', 'right of']

sentence_templates = [
    'a {size1} {colour1} {object1} is {relation} a {size2} {colour2} {object2}',
    'the {size1} {colour1} {object1} is positioned {relation} a {size2} {colour2} {object2}',
    'a {size1} {colour1} {object1} can be seen {relation} a {size2} {colour2} {object2}',
    '{relation} a {size2} {colour2} {object2} is a {size1} {colour1} {object1}',
]


def find_repo_root():
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / 'src').exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find repo root")


def build_candidates():
    candidates = []

    for template, o1, c1, s1, o2, c2, s2, rel in product(
        sentence_templates, objects, colours, sizes, objects, colours, sizes, relations
    ):
        if s1 == s2 and c1 == c2 and o1 == o2:
            continue

        sentence = template.format(
            size1=s1,
            colour1=c1,
            object1=o1,
            relation=rel,
            size2=s2,
            colour2=c2,
            object2=o2
        )

        candidates.append({
            'size1': s1,
            'colour1': c1,
            'object1': o1,
            'size2': s2,
            'colour2': c2,
            'object2': o2,
            'relation': rel,
            'sentence': sentence
        })

    return candidates


def main():
    root = find_repo_root()

    eps_folder = root / 'src' / 'data' / 'images' / 'type-a' / 'eps'
    data_folder = root / 'src' / 'data' / 'type-a'

    sentences_csv = data_folder / 'sentences_a.csv'
    master_csv = data_folder / 'master.csv'

    eps_folder.mkdir(parents=True, exist_ok=True)
    data_folder.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates()

    if NUM_OBSERVATIONS > len(candidates):
        print("Not enough unique combinations")
        return

    random.seed(RANDOM_SEED)
    observations = random.sample(candidates, NUM_OBSERVATIONS)

    screen = turtle.Screen()
    screen.setup(width=SCREEN_SIZE[0], height=SCREEN_SIZE[1])
    screen.tracer(0, 0)

    shape = ShapesGenerator()

    with open(sentences_csv, 'w', newline='', encoding='utf-8') as f1, \
         open(master_csv, 'w', newline='', encoding='utf-8') as f2:

        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)

        writer1.writerow(['sentence_id', 'sentence'])
        writer2.writerow(['path', 'label'])

        for i, obs in enumerate(observations):
            sentence_id = 'a_' + str(i)
            file_name = str(i) + '.eps'
            eps_path = eps_folder / file_name
            relative_path = Path('src/data/images/type-a/eps') / file_name

            shape.clear()
            shape.penup()
            shape.home()

            shape.draw_relation(
                obs['size1'],
                obs['colour1'],
                obs['object1'],
                obs['size2'],
                obs['colour2'],
                obs['object2'],
                obs['relation']
            )

            screen.update()
            screen.getcanvas().postscript(file=str(eps_path))

            writer1.writerow([sentence_id, obs['sentence']])
            writer2.writerow([str(relative_path).replace('\\', '/'), obs['sentence']])

            if (i + 1) % 500 == 0:
                print("Generated", i + 1, "/", NUM_OBSERVATIONS)

    print("sentences_a.csv created at:", sentences_csv)
    print("master.csv created at:", master_csv)
    print("Total observations generated:", NUM_OBSERVATIONS)

    turtle.bye()


if __name__ == '__main__':
    main()