import csv
import math
import random
from itertools import product
from pathlib import Path

from PIL import Image, ImageDraw

NUM_OBSERVATIONS = 10000
RANDOM_SEED = 42
CANVAS_SIZE = (700, 700)
BACKGROUND_COLOR = "white"

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

size_map = {
    "small": 45,
    "medium": 75,
    "big": 105,
}


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


def regular_polygon_points(cx, cy, radius, sides, rotation_degrees=0):
    points = []
    rotation_radians = math.radians(rotation_degrees)
    for i in range(sides):
        angle = rotation_radians + (2 * math.pi * i / sides)
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    return points


def draw_shape(draw, shape_name, cx, cy, size_label, colour):
    radius = size_map[size_label]

    if shape_name == "circle":
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=colour, outline=colour)

    elif shape_name == "square":
        draw.rectangle((cx - radius, cy - radius, cx + radius, cy + radius), fill=colour, outline=colour)

    elif shape_name == "triangle":
        points = regular_polygon_points(cx, cy, radius, 3, rotation_degrees=-90)
        draw.polygon(points, fill=colour, outline=colour)

    elif shape_name == "diamond":
        points = [
            (cx, cy - radius),
            (cx + radius, cy),
            (cx, cy + radius),
            (cx - radius, cy),
        ]
        draw.polygon(points, fill=colour, outline=colour)

    elif shape_name == "hexagon":
        points = regular_polygon_points(cx, cy, radius, 6, rotation_degrees=30)
        draw.polygon(points, fill=colour, outline=colour)

    elif shape_name == "octagon":
        points = regular_polygon_points(cx, cy, radius, 8, rotation_degrees=22.5)
        draw.polygon(points, fill=colour, outline=colour)

    else:
        raise ValueError(f"Unsupported shape: {shape_name}")


def get_positions(relation):
    center_x = CANVAS_SIZE[0] // 2
    center_y = CANVAS_SIZE[1] // 2
    gap = 180

    if relation == "left of":
        return (center_x - gap, center_y), (center_x + gap, center_y)
    elif relation == "right of":
        return (center_x + gap, center_y), (center_x - gap, center_y)
    elif relation == "above":
        return (center_x, center_y - gap), (center_x, center_y + gap)
    elif relation == "below":
        return (center_x, center_y + gap), (center_x, center_y - gap)
    else:
        raise ValueError(f"Unsupported relation: {relation}")


def create_image(obs, out_path):
    image = Image.new("RGB", CANVAS_SIZE, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)

    (x1, y1), (x2, y2) = get_positions(obs["relation"])

    draw_shape(draw, obs["object1"], x1, y1, obs["size1"], obs["colour1"])
    draw_shape(draw, obs["object2"], x2, y2, obs["size2"], obs["colour2"])

    image.save(out_path, format="PNG")


def main():
    print("PIL dataset generation started")

    root = find_repo_root()

    png_folder = root / 'src' / 'data' / 'images' / 'type-a' / 'png'
    data_folder = root / 'src' / 'data' / 'type-a'

    sentences_csv = data_folder / 'sentences_a.csv'
    master_csv = data_folder / 'master.csv'

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

    with open(sentences_csv, 'w', newline='', encoding='utf-8') as f1, \
         open(master_csv, 'w', newline='', encoding='utf-8') as f2:

        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)

        writer1.writerow(['sentence_id', 'sentence'])
        writer2.writerow(['path', 'label'])

        for i, obs in enumerate(observations):
            sentence_id = 'a_' + str(i)
            png_name = str(i) + '.png'
            png_path = png_folder / png_name
            relative_png_path = Path('src/data/images/type-a/png') / png_name

            create_image(obs, png_path)

            writer1.writerow([sentence_id, obs['sentence']])
            writer2.writerow([str(relative_png_path).replace('\\', '/'), obs['sentence']])

            if (i + 1) % 100 == 0 or (i + 1) == NUM_OBSERVATIONS:
                print("Generated", i + 1, "/", NUM_OBSERVATIONS)

    print("sentences_a.csv created at:", sentences_csv)
    print("master.csv created at:", master_csv)
    print("Total observations generated:", NUM_OBSERVATIONS)


if __name__ == '__main__':
    main()