# test_image_generator.py
from PIL import Image, ImageDraw
import numpy as np

def create_test_image(filename="test_microscopy.png", virus_like=True):
    img = Image.new('RGB', (224, 224), color=(245, 250, 248))
    draw = ImageDraw.Draw(img)
    
    if virus_like:
        # Viral kümeler (dairesel, kırmızımsı)
        for _ in range(15):
            cx, cy = np.random.randint(20, 204, 2)
            r = np.random.randint(8, 20)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(180, 60, 80))
    else:
        # Normal doku (düzenli hücreler)
        for row in range(0, 224, 28):
            for col in range(0, 224, 28):
                draw.ellipse([col-12, row-12, col+12, row+12], 
                           fill=(np.random.randint(210,235),)*3)
    
    img.save(filename)
    print(f"✅ Oluşturuldu: {filename}")

# Kullanım:
create_test_image("virus_test.png", virus_like=True)   # Hantavirus benzeri
create_test_image("normal_test.png", virus_like=False) # Normal doku benzeri