# check_happy_images.py
import os
from PIL import Image
import matplotlib.pyplot as plt

# Check what's actually in the happy folder
happy_dir = 'data/train_organized/happy'
happy_images = os.listdir(happy_dir)[:5]  # First 5 images

print(f"Sample happy images: {happy_images}")

# Visually inspect them
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, img_name in enumerate(happy_images):
    img = Image.open(os.path.join(happy_dir, img_name))
    axes[i].imshow(img)
    axes[i].set_title(img_name)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('verify_happy_images.png')
print("✅ Saved: verify_happy_images.png - Check if these are actually HAPPY faces!")