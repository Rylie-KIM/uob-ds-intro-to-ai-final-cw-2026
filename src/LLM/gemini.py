import os
from openai import OpenAI
from pathlib import Path
import base64

class Gemini:
    def __init__(self, img_name: int):
        self._ROOT = Path(__file__).resolve().parent.parent.parent

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        self.img_name = img_name
        self.prompt = (
            'I need you to generate a sentence capturing what is in this image. '
            'You must follow the instructions perfectly or you will fail. '
            'Use exactly one sentence template from the list below to structure your answer and DO NOT modify it.\n\n'
            'sentence_templates:\n'
            '1. a {size1} {colour1} {object1} is {relation} a {size2} {colour2} {object2}\n'
            '2. the {size1} {colour1} {object1} is positioned {relation} a {size2} {colour2} {object2}\n'
            '3. a {size1} {colour1} {object1} can be seen {relation} a {size2} {colour2} {object2}\n'
            '4. {relation} a {size2} {colour2} {object2} is a {size1} {colour1} {object1}\n\n'
            'Only use relations, objects, sizes and colours listed below.\n'
            'objects = [circle, triangle, square, diamond, hexagon, octagon].\n'
            'colours = [red, blue, green, yellow, orange, purple, pink, black].\n'
            'sizes = [big, medium, small].\n'
            'relations = [above, below, left of, right of].\n\n'
            'Do not add a period at the end.\n'
            'Do not use uppercase letters.\n'
            'Do not include any greetings, explanations, information about the background, or additional text not in the template.\n'
            'Output the final sentence using one template and no additional text.\n'
            'Do not comment on the background.'
        )

        self.completion = self.client.chat.completions.create(
            model="google/gemini-2.0-flash-lite-001",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.encode_img(self.img_name)
                            }
                        }
                    ]
                }
            ]
        )

    def encode_img(self, img_name: int):
        path = self._ROOT / 'src' / 'data' / 'images' / 'type-a' / 'png'
        full = path / f'{img_name}.png'
        if not full.exists():
            raise FileNotFoundError(f'Image path not found: {full}')
        processed = base64.b64encode(full.read_bytes()).decode("utf-8")
        return f'data:image/png;base64,{processed}'

    def get_output(self):
        output = self.completion.choices[0].message.content
        return output.strip()


if __name__ == "__main__":
    gem = Gemini(1)
    print(gem.get_output())
