import csv
import os
import random
import subprocess
import turtle
from itertools import product
from pathlib import Path

from relation_shapes_generator_a import ShapesGenerator

NUM_OBSERVATIONS = 50
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


def convert_eps_to_png(eps_path, png_path, imagemagick_cmd="magick"):
    if not eps_path.exists():
        raise FileNotFoundError(f"EPS file not found: {eps_path}")

    png_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        imagemagick_cmd,
        "-density", "300",
        str(eps_path),
        str(png_path)
    ]

    print("Running conversion:", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except FileNotFoundError:
        raise RuntimeError(
            "ImageMagick not found. Install ImageMagick and make sure 'magick' is in PATH."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Conversion timed out for {eps_path.name}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert {eps_path.name} to PNG.\nstdout: {e.stdout}\nstderr: {e.stderr}"
        )


def main():
    print("Script started")

    root = find_repo_root()

    eps_folder = root / 'src' / 'data' / 'images' / 'type-a' / 'eps'
    png_folder = root / 'src' / 'data' / 'images' / 'type-a' / 'png'
    data_folder = root / 'src' / 'data' / 'type-a'

    sentences_csv = data_folder / 'sentences_a.csv'
    master_csv = data_folder / 'master.csv'

    eps_folder.mkdir(parents=True, exist_ok=True)
    png_folder.mkdir(parents=True, exist_ok=True)
    data_folder.mkdir(parents=True, exist_ok=True)

    print("Building candidates...")
    candidates = build_candidates()
    print("Total candidate combinations:", len(candidates))

    if NUM_OBSERVATIONS > len(candidates):
        print("Not enough unique combinations")
        return

    random.seed(RANDOM_SEED)
    observations = random.sample(candidates, NUM_OBSERVATIONS)

    print("Creating turtle screen...")
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

            eps_name = str(i) + '.eps'
            png_name = str(i) + '.png'

            eps_path = eps_folder / eps_name
            png_path = png_folder / png_name
            relative_png_path = Path('src/data/images/type-a/png') / png_name

            print(f"Drawing sample {i + 1}/{NUM_OBSERVATIONS}")

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

            print("Saving EPS:", eps_path)
            screen.getcanvas().postscript(file=str(eps_path))

            print("Converting to PNG:", png_path)
            convert_eps_to_png(eps_path, png_path)

            writer1.writerow([sentence_id, obs['sentence']])
            writer2.writerow([str(relative_png_path).replace('\\', '/'), obs['sentence']])

            if (i + 1) % 10 == 0 or (i + 1) == NUM_OBSERVATIONS:
                print("Generated", i + 1, "/", NUM_OBSERVATIONS)

    print("sentences_a.csv created at:", sentences_csv)
    print("master.csv created at:", master_csv)
    print("Total observations generated:", NUM_OBSERVATIONS)

    turtle.bye()


if __name__ == '__main__':
    main()