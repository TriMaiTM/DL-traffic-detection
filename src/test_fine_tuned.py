import os
import cv2
import argparse
import time
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Fine-tuned Vehicle Detection (Evaluation)")
    parser.add_argument("--video", type=str, default="data/input/test.mp4", help="Path to input video file")
    parser.add_argument("--output", type=str, default="data/output/fine_tuned.mp4", help="Path to save output video")
    parser.add_argument("--weights", type=str, default="runs/train/yolov8_sgd_cos_false/weights/best.pt", help="Path to fine-tuned weights best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Check if weights exist
    if not os.path.exists(args.weights):
        print(f"[ERROR] Fine-tuned weights not found at: {args.weights}")
        print("Please ensure you have completed training first.")
        return

    # Check if input video exists
    if not os.path.exists(args.video):
        print(f"[ERROR] Input video '{args.video}' not found.")
        print(f"Please place your test video at: {os.path.abspath(args.video)}")
        return

    print(f"Loading Fine-tuned model: {args.weights}...")
    model = YOLO(args.weights)

    print(f"Opening video file: {args.video}...")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file '{args.video}'.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        fps = 30
        total_frames = 100

    print(f"Video Properties: Resolution={width}x{height}, FPS={fps:.1f}, Total Frames={total_frames}")

    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Processing video using Fine-tuned model and saving to '{args.output}'...")
    frame_count = 0
    start_time = time.time()

    # Class ID mapping of our fine-tuned model
    # 0: car, 1: motor, 2: truck, 3: bus
    class_names = {0: "Car", 1: "Motorcycle", 2: "Truck", 3: "Bus"}
    colors = {
        0: (0, 255, 0),       # Green for Car
        1: (255, 0, 0),       # Blue for Motorcycle
        2: (0, 0, 255),       # Red for Truck
        3: (0, 255, 255)      # Yellow for Bus
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run inference (detect all classes since the model is already specialized)
        results = model(frame, conf=args.conf, iou=args.iou, verbose=False)

        # Draw detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Get confidence score
                conf_score = float(box.conf[0])
                # Get class ID
                class_id = int(box.cls[0])

                color = colors.get(class_id, (255, 255, 255))
                label = f"{class_names.get(class_id, 'Vehicle')} {conf_score:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

                # Write label text
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Write the annotated frame
        out.write(frame)

        if frame_count % 30 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_time
            fps_proc = frame_count / elapsed
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Processed frame {frame_count}/{total_frames} ({progress:.1f}%) | Speed: {fps_proc:.1f} FPS")

    # Release resources
    cap.release()
    out.release()
    
    total_time = time.time() - start_time
    print(f"\nFinished processing in {total_time:.1f} seconds.")
    print(f"Fine-tuned model output saved to: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
