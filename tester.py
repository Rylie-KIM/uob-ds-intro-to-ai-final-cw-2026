import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_size = 100

path = r'C:\Users\fergu\Documents\GitHub\uob-ds-intro-to-ai-final-cw-2026\src\data\images\type-a\a_0.png'

img = Image.open(path).convert('RGB')
resize  = transforms.Resize((img_size, img_size))
img_resized = resize(img)

np_img = np.array(img_resized)

# print(np_img.shape)
plt.imshow(np_img)
plt.axis('off')
plt.show()

