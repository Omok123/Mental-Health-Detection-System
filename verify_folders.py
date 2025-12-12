# verify_folders.py
import os
from PIL import Image
import matplotlib.pyplot as plt

# Check what's in the 'happy' folder
happy_dir = 'data/train_organized/happy'
happy_images = sorted(os.listdir(happy_dir))[:5]

print("First 5 images in 'happy' folder:")
for img in happy_images:
    print(f"  {img}")

# Visually inspect them
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, img_name in enumerate(happy_images):
    img = Image.open(os.path.join(happy_dir, img_name))
    axes[i].imshow(img)
    axes[i].set_title(img_name[:15])
    axes[i].axis('off')
plt.suptitle("Images in 'happy' folder - ARE THESE HAPPY FACES?", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('verify_happy_folder.png', dpi=150)
print("\n✅ Saved: verify_happy_folder.png")
print("👁️  MANUALLY CHECK: Are these faces actually HAPPY?")