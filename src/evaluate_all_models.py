import os
from ultralytics import YOLO

def get_model_metrics(model_path, config_path="configs/traffic.yaml"):
    if not os.path.exists(model_path):
        print(f"[WARNING] Model path {model_path} does not exist.")
        return None
    
    print(f"\nEvaluating model: {model_path}...")
    model = YOLO(model_path)
    
    # Run validation on the dataset using the specified config
    # workers=0 to avoid Windows multiprocessing issues
    results = model.val(data=config_path, split="val", workers=0, verbose=False)
    
    # Extract class names and metrics
    names = results.names
    
    # Class-wise metrics
    # p, r, ap50, ap correspond to class-wise Precision, Recall, mAP50, mAP50-95
    p_class = results.box.p.tolist()
    r_class = results.box.r.tolist()
    ap50_class = results.box.ap50.tolist()
    ap_class = results.box.ap.tolist()
    
    class_metrics = {}
    for i, name in names.items():
        p = p_class[i]
        r = r_class[i]
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        class_metrics[name] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "map50": ap50_class[i],
            "map50_95": ap_class[i]
        }
        
    # Overall metrics
    overall_p = results.box.mp
    overall_r = results.box.mr
    overall_f1 = 2 * (overall_p * overall_r) / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    overall_map50 = results.box.map50
    overall_map50_95 = results.box.map
    
    class_metrics["all"] = {
        "precision": overall_p,
        "recall": overall_r,
        "f1": overall_f1,
        "map50": overall_map50,
        "map50_95": overall_map50_95
    }
    
    return class_metrics

def main():
    models = {
        "SGD + No Cosine": "runs/train/yolov8_sgd_cos_false/weights/best.pt",
        "Adam + No Cosine": "runs/train/yolov8_adam_cos_false/weights/best.pt",
        "SGD + Cosine Decay": "runs/train/yolov8_sgd_cos_true/weights/best.pt"
    }
    
    all_results = {}
    for name, path in models.items():
        metrics = get_model_metrics(path)
        if metrics:
            all_results[name] = metrics
            
    if not all_results:
        print("[ERROR] No models were successfully evaluated.")
        return
        
    classes = ["car", "motor", "truck", "bus", "all"]
    
    print("\n" + "="*80)
    print("DETAILED COMPREHENSIVE METRICS COMPARISON (VALIDATION SET)")
    print("="*80)
    
    for cls in classes:
        print(f"\n--- CLASS: {cls.upper()} ---")
        print(f"{'Model':<20}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'mAP@0.5':<12}{'mAP@0.5:0.95':<12}")
        print("-"*80)
        for model_name, metrics in all_results.items():
            m = metrics[cls]
            print(f"{model_name:<20}{m['precision']:<12.3f}{m['recall']:<12.3f}{m['f1']:<12.3f}{m['map50']:<12.3f}{m['map50_95']:<12.3f}")
            
    # Print a markdown table for easy copy-pasting
    print("\n" + "="*80)
    print("MARKDOWN TABLES FOR DOCS")
    print("="*80)
    
    for cls in classes:
        print(f"\n### Class: {cls.upper()}")
        print(f"| Model | Precision | Recall | F1-Score | mAP@0.5 | mAP@0.5:0.95 |")
        print(f"| :--- | :---: | :---: | :---: | :---: | :---: |")
        for model_name, metrics in all_results.items():
            m = metrics[cls]
            print(f"| **{model_name}** | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['map50']:.3f} | {m['map50_95']:.3f} |")

if __name__ == "__main__":
    main()
