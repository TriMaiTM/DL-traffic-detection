import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Fine-tuning (Phase 2 & 3)")
    parser.add_argument("--config", type=str, default="configs/traffic.yaml", help="Path to data config yaml file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (use smaller batch like 4 or 8 if out of memory)")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam", "AdamW", "auto"], help="Optimizer")
    parser.add_argument("--cos-lr", action="store_true", help="Use Cosine Learning Rate Scheduler (Cosine Decay)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. cpu, cuda, or 0)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model weights to start training from")
    parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers (0 is safest on Windows)")
    args = parser.parse_args()

    # Verify config file exists
    if not os.path.exists(args.config):
        print(f"[ERROR] Configuration file '{args.config}' not found.")
        return

    # Load model
    print(f"\nLoading pretrained model: {args.model}...")
    model = YOLO(args.model)

    # Detect if GPU is available if device is not specified
    import torch
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Start training
    print("-" * 60)
    print(f"Starting training on config: {args.config}")
    print(f"Parameters:")
    print(f" - Epochs: {args.epochs}")
    print(f" - Batch Size: {args.batch}")
    print(f" - Initial Learning Rate (lr0): {args.lr}")
    print(f" - Optimizer: {args.optimizer}")
    print(f" - Cosine LR Decay: {args.cos_lr}")
    print("-" * 60)

    # Train
    model.train(
        data=args.config,
        epochs=args.epochs,
        batch=args.batch,
        lr0=args.lr,
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
        device=device,
        workers=args.workers,
        project=os.path.abspath("runs/train"),
        name=f"yolov8_{args.optimizer.lower()}_cos_{str(args.cos_lr).lower()}",
        exist_ok=True
    )
    
    print("\nTraining completed successfully!")
    print(f"Results and weights are saved in runs/train/yolov8_{args.optimizer.lower()}_cos_{str(args.cos_lr).lower()}")

if __name__ == "__main__":
    main()
