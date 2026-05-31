import os
import cv2
import numpy as np
from ultralytics import YOLO

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
    print("Initializing Base Model (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")
    
    images_dir = "data/dataset/valid/images"
    labels_dir = "data/dataset/valid/labels"
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("[ERROR] Validation dataset directories not found.")
        return

    # COCO Class mapping to Custom indices
    # COCO: 2 -> car, 3 -> motorcycle (motor), 7 -> truck, 5 -> bus
    coco_to_custom = {2: 0, 3: 1, 7: 2, 5: 3}
    class_names = {0: "car", 1: "motor", 2: "truck", 3: "bus"}
    
    # Store TP, FP, FN per class
    # class_id: [TP, FP, FN]
    stats = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Evaluating base model on {len(image_files)} validation images...")
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        lbl_file = os.path.splitext(img_file)[0] + ".txt"
        lbl_path = os.path.join(labels_dir, lbl_file)
        
        # 1. Read Ground Truth
        gt_boxes = {0: [], 1: [], 2: [], 3: []}
        if os.path.exists(lbl_path):
            # Read image dimensions
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    if cls_id in gt_boxes:
                        # Convert normalized xywh to absolute xyxy
                        x_c, y_c, box_w, box_h = map(float, parts[1:5])
                        x1 = (x_c - box_w/2) * w
                        y1 = (y_c - box_h/2) * h
                        x2 = (x_c + box_w/2) * w
                        y2 = (y_c + box_h/2) * h
                        gt_boxes[cls_id].append([x1, y1, x2, y2])
                        
        # 2. Get Predictions from COCO model
        # Use conf=0.25 (standard confidence threshold)
        results = model(img_path, conf=0.25, iou=0.7, verbose=False)[0]
        pred_boxes = {0: [], 1: [], 2: [], 3: []}
        
        for box in results.boxes:
            coco_cls = int(box.cls[0])
            if coco_cls in coco_to_custom:
                custom_cls = coco_to_custom[coco_cls]
                xyxy = box.xyxy[0].tolist()
                pred_boxes[custom_cls].append(xyxy)
                
        # 3. Calculate TP, FP, FN for each class
        for cls_id in [0, 1, 2, 3]:
            preds = pred_boxes[cls_id]
            gts = gt_boxes[cls_id]
            
            matched_gts = set()
            tps = 0
            fps = 0
            
            for pred in preds:
                best_iou = 0
                best_gt_idx = -1
                for idx, gt in enumerate(gts):
                    if idx in matched_gts:
                        continue
                    iou = calculate_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                        
                if best_iou >= 0.5:
                    tps += 1
                    matched_gts.add(best_gt_idx)
                else:
                    fps += 1
                    
            fns = len(gts) - len(matched_gts)
            
            stats[cls_id][0] += tps
            stats[cls_id][1] += fps
            stats[cls_id][2] += fns

    # Print Results
    print("\n" + "="*50)
    print(" BASE PRE-TRAINED MODEL (COCO) EVALUATION METRICS")
    print("="*50)
    print(f"{'Class':<12}{'Precision':<15}{'Recall':<15}")
    print("-"*50)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for cls_id in [0, 1, 2, 3]:
        tp, fp, fn = stats[cls_id]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"{class_names[cls_id]:<12}{precision:<15.3f}{recall:<15.3f}")
        
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    print("-"*50)
    print(f"{'all':<12}{overall_prec:<15.3f}{overall_rec:<15.3f}")
    print("="*50)
    
    # Print comparison with fine-tuned model (from user runs)
    print("\nCOMPARISON TABLE: BASE (COCO) VS FINE-TUNED (SGD)")
    print("="*65)
    print(f"{'Class':<12}{'Base Prec':<12}{'Base Rec':<12}{'FT Prec':<12}{'FT Rec':<12}")
    print("-"*65)
    
    # Fine-tuned stats (user's run output)
    ft_stats = {
        0: (0.872, 0.898),  # car
        1: (0.853, 0.813),  # motor
        2: (0.734, 0.879),  # truck
        3: (0.892, 0.771),  # bus
        'all': (0.838, 0.840)
    }
    
    for cls_id in [0, 1, 2, 3]:
        tp, fp, fn = stats[cls_id]
        bp = tp / (tp + fp) if (tp + fp) > 0 else 0
        br = tp / (tp + fn) if (tp + fn) > 0 else 0
        ftp, ftr = ft_stats[cls_id]
        print(f"{class_names[cls_id]:<12}{bp:<12.3f}{br:<12.3f}{ftp:<12.3f}{ftr:<12.3f}")
        
    print("-"*65)
    all_ftp, all_ftr = ft_stats['all']
    print(f"{'all':<12}{overall_prec:<12.3f}{overall_rec:<12.3f}{all_ftp:<12.3f}{all_ftr:<12.3f}")
    print("="*65)

if __name__ == "__main__":
    main()
