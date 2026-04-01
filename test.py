import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

path = r'C:\Users\fergu\Documents\GitHub\uob-ds-intro-to-ai-final-cw-2026\src\data\type-a\type-a-dataset\type-a-dataset-png\1.png'
img = Image.open(path).convert('RGB')
# img.show()
np_img = np.array(img)
print(np_img.shape)
plt.imshow(np_img)
plt.axis('off')
plt.show()