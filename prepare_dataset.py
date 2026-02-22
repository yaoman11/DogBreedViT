# Script untuk gabung dataset + crawled_images, crop pake annotation, terus split
import os
import shutil
import random
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path
from collections import defaultdict

# mapping nama folder ke class name yang bersih
FOLDER_TO_CLASS = {
    # dari folder dataset
    "1160-n000003-Siberian_husky": "Siberian_Husky",
    "206-n000035-Great_Dane": "Great_Dane",
    "209-n000054-Doberman": "Doberman",
    "211-n000018-German_shepherd": "German_Shepherd",
    "211-n000061-Bernese_mountain_dog": "Bernese_Mountain_Dog",
    "2192-n000088-Samoyed": "Samoyed",
    "224-n000060-Rottweiler": "Rottweiler",
    "225-n000082-boxer": "Boxer",
    "3580-n000122-Labrador_retriever": "Labrador_Retriever",
    "5355-n000126-golden_retriever": "Golden_Retriever",
    # dari folder crawled_images (udah bersih)
    "Bernese_Mountain_Dog": "Bernese_Mountain_Dog",
    "Boxer": "Boxer",
    "Doberman": "Doberman",
    "German_Shepherd": "German_Shepherd",
    "Great_Dane": "Great_Dane",
    "Rottweiler": "Rottweiler",
    "Samoyed": "Samoyed",
    "Siberian_Husky": "Siberian_Husky",
    "Labrador_Retriever": "Labrador_Retriever",
    "Golden_Retriever": "Golden_Retriever",
}

DATASET_DIR = "dataset"
ANNOTATION_DIR = "annotation"
CRAWLED_DIR = "crawled_images"
OUTPUT_DIR = "data_merged"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def parse_annotation(xml_path):
    """Parse XML annotation, ambil bodybndbox"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        obj = root.find("object")
        if obj is None:
            return None
        
        bbox = obj.find("bodybndbox")
        if bbox is None:
            return None
        
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        
        return (xmin, ymin, xmax, ymax)
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None


def crop_with_padding(img, bbox, padding=0.1):
    """Crop image dengan padding biar ga kepotong"""
    w, h = img.size
    xmin, ymin, xmax, ymax = bbox
    
    bw = xmax - xmin
    bh = ymax - ymin
    px = int(bw * padding)
    py = int(bh * padding)
    
    xmin = max(0, xmin - px)
    ymin = max(0, ymin - py)
    xmax = min(w, xmax + px)
    ymax = min(h, ymax + py)
    
    return img.crop((xmin, ymin, xmax, ymax))


def process_dataset_folder():
    """Process folder dataset - crop pake annotation"""
    images = []
    
    for folder_name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        class_name = FOLDER_TO_CLASS.get(folder_name)
        if not class_name:
            print(f"Skip unknown folder: {folder_name}")
            continue
        
        annotation_folder = os.path.join(ANNOTATION_DIR, folder_name)
        has_annotations = os.path.exists(annotation_folder)
        
        for img_file in os.listdir(folder_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(folder_path, img_file)
            
            # cek ada annotation ga
            bbox = None
            if has_annotations:
                xml_file = img_file + ".xml"
                xml_path = os.path.join(annotation_folder, xml_file)
                if os.path.exists(xml_path):
                    bbox = parse_annotation(xml_path)
            
            images.append({
                'path': img_path,
                'class': class_name,
                'bbox': bbox,
                'source': 'dataset'
            })
    
    return images


def process_crawled_folder():
    """Process folder crawled_images - langsung pake (ga perlu crop)"""
    images = []
    
    for folder_name in os.listdir(CRAWLED_DIR):
        folder_path = os.path.join(CRAWLED_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        class_name = FOLDER_TO_CLASS.get(folder_name)
        if not class_name:
            print(f"Skip unknown folder: {folder_name}")
            continue
        
        for img_file in os.listdir(folder_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            
            img_path = os.path.join(folder_path, img_file)
            images.append({
                'path': img_path,
                'class': class_name,
                'bbox': None,  # ga perlu crop
                'source': 'crawled'
            })
    
    return images


def save_image(img_info, output_path, idx):
    """Save image - crop kalo ada bbox, copy kalo ga ada"""
    try:
        img = Image.open(img_info['path'])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # crop kalo ada annotation
        if img_info['bbox']:
            try:
                img = crop_with_padding(img, img_info['bbox'])
            except Exception:
                pass  # skip crop kalo gagal, pake original
        
        # save dengan nama baru
        ext = Path(img_info['path']).suffix
        if ext.lower() == '.webp':
            ext = '.jpg'
        
        filename = f"{img_info['class']}_{idx:04d}{ext}"
        save_path = os.path.join(output_path, filename)
        img.save(save_path, quality=95)
        return True
    except Exception as e:
        return False


def split_and_save(all_images):
    """Split data dan save ke train/val/test folders"""
    # group by class
    class_images = defaultdict(list)
    for img in all_images:
        class_images[img['class']].append(img)
    
    # buat output folders
    for split in ['train', 'val', 'test']:
        for class_name in set(FOLDER_TO_CLASS.values()):
            os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)
    
    stats = {'train': 0, 'val': 0, 'test': 0}
    class_stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'cropped': 0})
    
    total = len(all_images)
    processed = 0
    
    for class_name, images in class_images.items():
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train+n_val],
            'test': images[n_train+n_val:]
        }
        
        for split_name, split_images in splits.items():
            output_folder = os.path.join(OUTPUT_DIR, split_name, class_name)
            
            for idx, img_info in enumerate(split_images):
                if save_image(img_info, output_folder, idx):
                    stats[split_name] += 1
                    class_stats[class_name][split_name] += 1
                    if img_info['bbox']:
                        class_stats[class_name]['cropped'] += 1
                
                processed += 1
                if processed % 500 == 0:
                    print(f"    Progress: {processed}/{total} ({processed*100//total}%)")
    
    return stats, class_stats


def main():
    print("=" * 50)
    print("PREPARE DATASET: Merge + Crop + Split")
    print("=" * 50)
    
    random.seed(42)  # biar reproducible
    
    # collect semua images
    print("\n[1] Processing dataset folder...")
    dataset_images = process_dataset_folder()
    print(f"    Found {len(dataset_images)} images")
    cropped_count = sum(1 for img in dataset_images if img['bbox'])
    print(f"    {cropped_count} images have annotations (will be cropped)")
    
    print("\n[2] Processing crawled_images folder...")
    crawled_images = process_crawled_folder()
    print(f"    Found {len(crawled_images)} images")
    
    all_images = dataset_images + crawled_images
    print(f"\n[3] Total: {len(all_images)} images")
    
    # split dan save
    print(f"\n[4] Splitting and saving to '{OUTPUT_DIR}/'...")
    stats, class_stats = split_and_save(all_images)
    
    # summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Train: {stats['train']} images")
    print(f"Val:   {stats['val']} images")
    print(f"Test:  {stats['test']} images")
    print(f"Total: {sum(stats.values())} images")
    
    print("\nPer class:")
    print(f"{'Class':<25} {'Train':>6} {'Val':>6} {'Test':>6} {'Cropped':>8}")
    print("-" * 55)
    for class_name in sorted(class_stats.keys()):
        s = class_stats[class_name]
        print(f"{class_name:<25} {s['train']:>6} {s['val']:>6} {s['test']:>6} {s['cropped']:>8}")
    
    print(f"\nOutput saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
