import os
from ultralytics import YOLO
import numpy as np

def calculate_iou(box1, box2):
    # box = [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    return intersection / union

def main():
    model = YOLO("yolov8n.pt")
    
    images_dir = "vietnamese_vehicle/train/images"
    labels_dir = "vietnamese_vehicle/train/labels"
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("[ERROR] Dataset directory not found.")
        return

    # COCO Class mapping: 2 -> car, 3 -> motorcycle/motor, 5 -> bus, 7 -> truck
    coco_names = {2: "car", 3: "motor", 5: "bus", 7: "truck"}
    
    # Store matches
    # key: custom_class_id, value: dict of {coco_class_name: count}
    matches = {0: {}, 1: {}, 2: {}, 3: {}}
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:200] # Check first 200 images
    
    print(f"Analyzing {len(image_files)} images to reverse-engineer class mapping...")
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        lbl_file = os.path.splitext(img_file)[0] + ".txt"
        lbl_path = os.path.join(labels_dir, lbl_file)
        
        if not os.path.exists(lbl_path):
            continue
            
        # Get predictions from pre-trained model
        results = model(img_path, conf=0.3, verbose=False)[0]
        pred_boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in coco_names:
                pred_boxes.append({
                    'box': box.xyxy[0].tolist(),
                    'class_name': coco_names[cls_id]
                })
                
        # Read dataset labels
        h, w = results.orig_shape[:2]
        custom_boxes = []
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                # Convert normalized xywh to absolute xyxy
                x_c, y_c, box_w, box_h = map(float, parts[1:5])
                x1 = (x_c - box_w/2) * w
                y1 = (y_c - box_h/2) * h
                x2 = (x_c + box_w/2) * w
                y2 = (y_c + box_h/2) * h
                custom_boxes.append({
                    'class_id': cls_id,
                    'box': [x1, y1, x2, y2]
                })
                
        # Match boxes using IoU
        for cb in custom_boxes:
            best_iou = 0
            best_match = None
            for pb in pred_boxes:
                iou = calculate_iou(cb['box'], pb['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = pb['class_name']
                    
            if best_iou > 0.5 and best_match is not None:
                c_id = cb['class_id']
                matches[c_id][best_match] = matches[c_id].get(best_match, 0) + 1

    print("\nConsensus Class Mapping Results:")
    print("-" * 50)
    final_mapping = {}
    for c_id in sorted(matches.keys()):
        counts = matches[c_id]
        if counts:
            best_fit = max(counts, key=counts.get)
            total = sum(counts.values())
            pct = (counts[best_fit] / total) * 100
            print(f"Custom Index {c_id} -> Maps to '{best_fit}' (Confidence: {pct:.1f}% based on {total} matches)")
            print(f"  Breakdown: {counts}")
            final_mapping[c_id] = best_fit
        else:
            print(f"Custom Index {c_id} -> No matches found")
            final_mapping[c_id] = f"unknown_{c_id}"
    print("-" * 50)
    
    # Generate proposed configs/traffic.yaml content
    print("\nProposed configs/traffic.yaml names list:")
    print("names:")
    for c_id in sorted(final_mapping.keys()):
        print(f"  {c_id}: {final_mapping[c_id]}")

if __name__ == "__main__":
    main()
