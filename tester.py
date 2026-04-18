import torch
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
sns.set_style('darkgrid')
import torch.nn as nn
from pathlib import Path
"""
path = Path("C:/Masters/Text Analytics/AI_Coursework/trained_models/GoogleNet_sbert.pt")
model = torch.load(path, map_location=torch.device('cpu'))
train_loss = model['train_losses']
val_loss = model['val_losses']
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.show()
"""