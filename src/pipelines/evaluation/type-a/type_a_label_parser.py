import re


def parse_label(text):
    text = str(text).strip().lower()

    patterns = [
        r"a (\w+) (\w+) (\w+) is (above|below|left of|right of) a (\w+) (\w+) (\w+)",
        r"the (\w+) (\w+) (\w+) is positioned (above|below|left of|right of) a (\w+) (\w+) (\w+)",
        r"a (\w+) (\w+) (\w+) can be seen (above|below|left of|right of) a (\w+) (\w+) (\w+)",
        r"(above|below|left of|right of) a (\w+) (\w+) (\w+) is a (\w+) (\w+) (\w+)",
    ]

    for idx, pattern in enumerate(patterns):
        match = re.fullmatch(pattern, text)
        if match:
            groups = match.groups()

            if idx in [0, 1, 2]:
                return {
                    "size1": groups[0],
                    "colour1": groups[1],
                    "object1": groups[2],
                    "relation": groups[3],
                    "size2": groups[4],
                    "colour2": groups[5],
                    "object2": groups[6],
                }
            else:
                return {
                    "relation": groups[0],
                    "size2": groups[1],
                    "colour2": groups[2],
                    "object2": groups[3],
                    "size1": groups[4],
                    "colour1": groups[5],
                    "object1": groups[6],
                }

    return None