import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIT_MODEL_PATH = "vit_merged_best_model.pth"
YOLO_MODEL = "yolov8n.pt"

# Threshold detection
YOLO_CONF_THRESHOLD = 0.4
BREED_CONF_THRESHOLD = 0.85

CLASS_NAMES = [
    "Bernese_Mountain_Dog", "Boxer", "Doberman", "German_Shepherd",
    "Golden_Retriever", "Great_Dane", "Labrador_Retriever",
    "Rottweiler", "Samoyed", "Siberian_Husky"
]


def load_yolo_model(model_path=YOLO_MODEL):
    from ultralytics import YOLO
    print(f"Loading YOLO: {model_path}")
    return YOLO(model_path)


def load_vit_model(model_path=VIT_MODEL_PATH, num_classes=10):
    from models.vit_model import build_vit_model
    print(f"Loading ViT: {model_path}")
    model = build_vit_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def detect_dogs_yolo(yolo_model, image_path, conf_thresh=YOLO_CONF_THRESHOLD):
    results = yolo_model(image_path, verbose=False)
    detections = []
    
    for result in results:
        for box in result.boxes:
            # class 16 = dog
            if int(box.cls) == 16 and float(box.conf) >= conf_thresh:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(box.conf)
                })
    return detections


def crop_image(image, bbox, padding=0.1):
    w, h = image.size
    x1, y1, x2, y2 = bbox
    
    # add padding
    px = int((x2 - x1) * padding)
    py = int((y2 - y1) * padding)
    
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)
    
    return image.crop((x1, y1, x2, y2))


def classify_dog_breed(vit_model, cropped_img, min_confidence=BREED_CONF_THRESHOLD):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if cropped_img.mode != 'RGB':
        cropped_img = cropped_img.convert('RGB')
    
    img = preprocess(cropped_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out = vit_model(img).logits
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, idx = torch.max(probs, 1)
        
        conf_score = conf.item()
        pred_breed = CLASS_NAMES[idx.item()]
        all_probs = {CLASS_NAMES[i]: probs[0][i].item() for i in range(len(CLASS_NAMES))}
        
        is_valid = conf_score >= min_confidence
        breed = pred_breed if is_valid else "Unknown"
    
    return {
        'breed': breed,
        'predicted_breed': pred_breed,
        'confidence': conf_score,
        'is_valid': is_valid,
        'all_probs': all_probs
    }


def inference_pipeline(image_path, yolo_model, vit_model, save_result=True, 
                       yolo_conf=YOLO_CONF_THRESHOLD, breed_conf=BREED_CONF_THRESHOLD):
    print(f"\nProcessing: {image_path}")
    
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # detect dogs
    print("Detecting...")
    detections = detect_dogs_yolo(yolo_model, image_path, conf_thresh=yolo_conf)
    print(f"Found {len(detections)} object(s)")
    
    if not detections:
        print("No dogs detected")
        return []
    
    # classify each detection
    results = []
    valid_count = 0
    
    for i, det in enumerate(detections):
        print(f"Classifying #{i + 1}...")
        cropped = crop_image(img, det['bbox'])
        cls = classify_dog_breed(vit_model, cropped, min_confidence=breed_conf)
        
        result = {
            'dog_id': i + 1,
            'bbox': det['bbox'],
            'detection_confidence': det['confidence'],
            'breed': cls['breed'],
            'predicted_breed': cls['predicted_breed'],
            'breed_confidence': cls['confidence'],
            'is_valid': cls['is_valid'],
            'all_breed_probs': cls['all_probs']
        }
        results.append(result)
        
        if cls['is_valid']:
            print(f"  -> {result['breed']} ({result['breed_confidence']:.1%})")
            valid_count += 1
        else:
            print(f"  -> Unknown")
    
    print(f"Valid: {valid_count}/{len(results)}")
    
    if save_result:
        visualize_results(img, results, image_path)
    
    return results


def visualize_results(image, results, original_path):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    colors = plt.cm.Set1(np.linspace(0, 1, 10))
    valid = sum(1 for r in results if r.get('is_valid', True))
    
    for r in results:
        x1, y1, x2, y2 = r['bbox']
        breed = r['breed']
        conf = r['breed_confidence']
        
        if r.get('is_valid', True):
            color = colors[CLASS_NAMES.index(breed) % 10]
            label = f"{breed}\n{conf:.1%}"
        else:
            color = (0.5, 0.5, 0.5, 1.0)
            label = "Unknown"
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                  edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        ax.text(x1, y1-5, label, color='white', fontsize=10, fontweight='bold',
                va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    
    ax.set_title(f"Detected: {len(results)} | Identified: {valid}", fontsize=12)
    ax.axis('off')
    
    out = Path(original_path).stem + "_result.png"
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {out}")


def main():
    print("=" * 50)
    print("Dog Breed Classification")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Thresholds - YOLO: {YOLO_CONF_THRESHOLD:.0%}, Breed: {BREED_CONF_THRESHOLD:.0%}")
    print("=" * 50)
    
    yolo = load_yolo_model()
    vit = load_vit_model()
    
    test_img = "download (7).jpg"
    
    if os.path.exists(test_img):
        results = inference_pipeline(test_img, yolo, vit)
        
        if results:
            print("\n" + "=" * 50)
            print("RESULTS")
            print("=" * 50)
            for r in results:
                if r['is_valid']:
                    print(f"#{r['dog_id']}: {r['breed']} ({r['breed_confidence']:.1%})")
                else:
                    print(f"#{r['dog_id']}: Unknown")
    else:
        print(f"\nFile not found: {test_img}")
        print("Usage: results = inference_pipeline('image.jpg', yolo, vit)")


if __name__ == "__main__":
    main()