import os
import cv2
import argparse
import time
from ultralytics import YOLO

def draw_live_panel(frame, counts):
    """
    Draws a premium, semi-transparent dashboard showing the current vehicle count
    in the top-left corner of the frame.
    """
    # Panel coordinates
    x1, y1 = 20, 20
    w, h = 250, 160
    
    # Create a semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + w, y1 + h), (40, 40, 40), -1)  # Dark charcoal background
    
    # Blend overlay with original frame (alpha = 0.7 for premium glassmorphism feel)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw a thin stylish border around the panel
    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (52, 152, 219), 2)  # Bright blue border
    
    # Title
    cv2.putText(frame, "LIVE TRAFFIC COUNTER", (x1 + 15, y1 + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 152, 219), 2, cv2.LINE_AA)
    cv2.line(frame, (x1 + 15, y1 + 40), (x1 + w - 15, y1 + 40), (80, 80, 80), 1)
    
    # Stats details
    class_names = {0: "Cars", 1: "Motors", 2: "Trucks", 3: "Buses"}
    colors = {
        0: (46, 204, 113),    # Green
        1: (52, 152, 219),    # Blue
        2: (231, 76, 60),     # Red
        3: (241, 196, 15)     # Yellow
    }
    
    curr_y = y1 + 65
    for cls_id, label in class_names.items():
        count = counts.get(cls_id, 0)
        color = colors.get(cls_id, (255, 255, 255))
        
        # Draw class color indicator dot
        cv2.circle(frame, (x1 + 25, curr_y - 5), 5, color, -1)
        
        # Text label
        cv2.putText(frame, f"{label}:", (x1 + 40, curr_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        
        # Text value
        cv2.putText(frame, f"{count}", (x1 + 180, curr_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        
        curr_y += 25

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Fine-tuned Vehicle Tracking (Phase 4)")
    parser.add_argument("--video", type=str, default="data/input/test.mp4", help="Path to input video file")
    parser.add_argument("--output", type=str, default="data/output/tracking.mp4", help="Path to save output video")
    parser.add_argument("--model", type=str, default="runs/train/yolov8_sgd_cos_true/weights/best.pt", help="Path to best model weights")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"], help="Tracker configuration file")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS/association (higher value allows overlapping crowded vehicles)")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size (e.g. 640, 960, 1080)")
    parser.add_argument("--roi-y", type=float, default=0.35, help="Top portion of the frame height to ignore (ROI). E.g., 0.35 ignores top 35% of the frame.")
    parser.add_argument("--show-roi", action="store_true", help="Show ROI boundary line on the output video")
    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"[ERROR] Fine-tuned model weights not found at: {args.model}")
        print("Please ensure Phase 3 training completed or specify another model path using --model.")
        return

    # Check if input video exists
    if not os.path.exists(args.video):
        print(f"[ERROR] Input video '{args.video}' not found.")
        print("Please place your test video inside 'data/input/test.mp4' or pass it using --video.")
        return

    print(f"Loading Fine-Tuned model: {args.model}...")
    model = YOLO(args.model)

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
        print("[WARNING] Could not retrieve FPS/total frames. Falling back to defaults.")

    print(f"Video Properties: Resolution={width}x{height}, FPS={fps:.1f}, Total Frames={total_frames}")

    # Set up VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Running tracking using {args.tracker} and writing output to '{args.output}'...")
    
    frame_count = 0
    start_time = time.time()

    # Mapped custom classes
    class_names = {0: "Car", 1: "Motor", 2: "Truck", 3: "Bus"}
    colors = {
        0: (46, 204, 113),    # Emerald Green for Car
        1: (52, 152, 219),    # Peter River Blue for Motor
        2: (231, 76, 60),     # Alizarin Red for Truck
        3: (241, 196, 15)     # Sun Flower Yellow for Bus
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Track objects in the frame
        # persist=True maintains tracks across frames
        results = model.track(
            source=frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False
        )[0]

        # Initialize counts for the current frame
        current_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        # Calculate ROI limit in pixels
        roi_y_limit = int(args.roi_y * height)
        if args.roi_y > 0 and args.show_roi:
            # Draw a premium semi-transparent ROI line
            cv2.line(frame, (0, roi_y_limit), (width, roi_y_limit), (231, 76, 60), 2, lineType=cv2.LINE_AA) # Red ROI Line
            cv2.putText(frame, "ROI BOUNDARY - ACTIVE SCANNING ZONE BELOW", (30, roi_y_limit - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (231, 76, 60), 1, cv2.LINE_AA)

        # Draw tracking lines and bounding boxes
        if results.boxes is not None:
            boxes = results.boxes
            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Apply ROI filter: ignore objects whose bottom coordinate (y2) is above the ROI limit
                if args.roi_y > 0 and y2 < roi_y_limit:
                    continue
                    
                # Class index
                cls_id = int(box.cls[0])
                # Confidence score
                conf_score = float(box.conf[0])
                
                # Get tracking ID if available
                track_id = int(box.id[0]) if box.id is not None else None
                
                # Increment count
                if cls_id in current_counts:
                    current_counts[cls_id] += 1

                # Colors & labels
                color = colors.get(cls_id, (255, 255, 255))
                class_label = class_names.get(cls_id, "Vehicle")
                
                if track_id is not None:
                    label = f"{class_label} #{track_id} {conf_score:.2f}"
                else:
                    label = f"{class_label} {conf_score:.2f}"

                # Draw smooth bounding box with semi-rounded edge feeling (double rectangle for thickness)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

                # Draw text background
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)

                # Draw label text
                cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw the live statistics overlay panel
        draw_live_panel(frame, current_counts)

        # Write annotated frame to output video
        out.write(frame)

        # Progress reporting
        if frame_count % 30 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_time
            fps_proc = frame_count / elapsed
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Tracking: Frame {frame_count}/{total_frames} ({progress:.1f}%) | Speed: {fps_proc:.1f} FPS")

    # Release resources
    cap.release()
    out.release()

    total_time = time.time() - start_time
    print(f"\nTracking completed in {total_time:.1f} seconds.")
    print(f"Annotated tracking video saved to: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
