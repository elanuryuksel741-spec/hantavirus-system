# prepare_malaria_dataset.py
import os
import shutil
import random

random.seed(42)
BASE_DIR = os.getcwd()
SRC_DIR = os.path.join(BASE_DIR, 'cell_images')
TARGET_DIR = os.path.join(BASE_DIR, 'real_data')

# Dataset -> Proje sınıfı eşleştirmesi
CLASS_MAP = {
    'Parasitized': 'hantavirus',
    'Uninfected': 'normal'
}

def prepare():
    # Hedef klasör yapısını oluştur
    splits = ['train', 'val', 'test']
    classes = ['hantavirus', 'normal']
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

    # Her sınıf için dosyaları tara, karıştır ve böl
    for src_folder, target_class in CLASS_MAP.items():
        src_path = os.path.join(SRC_DIR, src_folder)
        if not os.path.exists(src_path):
            print(f"❌ HATA: '{src_path}' bulunamadı. cell_images klasörünün proje kök dizininde olduğundan emin olun.")
            return

        # Sadece görsel dosyalarını al
        files = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"📁 {src_folder}: {len(files)} görsel bulundu.")
        random.shuffle(files)

        n = len(files)
        train_end = int(n * 0.70)
        val_end = train_end + int(n * 0.15)

        # Train / Val / Test kopyalama
        for split_name, start, end in [('train', 0, train_end), ('val', train_end, val_end), ('test', val_end, n)]:
            subset = files[start:end]
            dest_dir = os.path.join(TARGET_DIR, split_name, target_class)
            for f in subset:
                shutil.copy(os.path.join(src_path, f), os.path.join(dest_dir, f))
            print(f"   ↳ {split_name.upper()}: {len(subset)} görsel kopyalandı → {target_class}/")

    print("\n🎉 Dataset hazırlama tamamlandı! 'real_data/' yapısı doğrulandı.")

if __name__ == '__main__':
    prepare()