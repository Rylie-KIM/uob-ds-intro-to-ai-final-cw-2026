objects = ['circle','triangle','square','rectangle','diamond', ' star', 'pentagon', 'hexagon', 'octagon', 'cube']
colours = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink']
size = ['big', 'medium', 'small']
relations = ['above', 'below', 'left of', 'right of']
sentence_templates = [
    'a {bundle1} is {relation} a {bundle2}',
    'the {bundle1} is {relation} a {bundle2}',
    'a {bundle1} can be seen {relation} a {bundle2}',
    '{relation} a {bundle1} is a {bundle2}',
    'a {bundle1} can be seen {relation} a {bundle2}'
]

def creator(objects:[List], colours:[list], size:[list], sentence_templates:[list]) -> [list]:
    """
    Creates list of structured sentences where each bundle represents a concatenated str of <size> <colour> <object> 
    Output is list of structured sentences
    """
    sentences = []
    for template in sentence_templates:
        for object1 in objects:
            for colour1 in colours:
                for s1 in size:
                    for object2 in objects:
                        for colour2 in colours:
                            for s2 in size:
                                for rel in relations:
                                    
                                    bundle1 = f'{s1} {colour1} {object1}'
                                    relation = f'{rel}'
                                    bundle2 = f'{s2} {colour2} {object2}'
                                    if bundle1 == bundle2:
                                        continue
                                    sentences.append(template.format(bundle1=bundle1, relation=relation, bundle2=bundle2))
    return sentences


sentences = creator(objects, colours, size, sentence_templates)
print(len(sentences))
print(sentences[:5])