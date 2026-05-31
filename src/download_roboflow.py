import os
import argparse
import shutil
from roboflow import Roboflow

def main():
    parser = argparse.ArgumentParser(description="Download dataset from Roboflow Universe")
    parser.add_argument("--api_key", type=str, required=True, help="Your private Roboflow API Key")
    parser.add_argument("--workspace", type=str, default="traffic-sign-detector-2u2n9", help="Roboflow workspace name")
    parser.add_argument("--project", type=str, default="vietnamese-vehicles-detector", help="Roboflow project name")
    parser.add_argument("--version", type=int, default=1, help="Dataset version")
    args = parser.parse_args()

    # Initialize Roboflow
    print(f"Connecting to Roboflow using API Key...")
    try:
        rf = Roboflow(api_key=args.api_key)
        workspace = rf.workspace(args.workspace)
        project = workspace.project(args.project)
    except Exception as e:
        print(f"\n[ERROR] Authentication failed. Please check your API key.")
        print(f"Details: {e}")
        return

    # Download dataset
    print(f"Downloading project '{args.project}' (v{args.version}) in YOLOv8 format...")
    try:
        dataset = project.version(args.version).download("yolov8")
    except Exception as e:
        print(f"\n[ERROR] Failed to download dataset. Check project name or version.")
        print(f"Details: {e}")
        return

    print(f"Download completed. Location: {dataset.location}")

    # Reorganize downloaded folder to data/dataset
    target_dir = "data/dataset"
    if os.path.exists(target_dir):
        print(f"Removing old dataset directory at '{target_dir}'...")
        shutil.rmtree(target_dir)

    print(f"Moving dataset to '{target_dir}'...")
    shutil.move(dataset.location, target_dir)

    # Clean up empty download folder if it remains
    if os.path.exists(dataset.location):
        shutil.rmtree(dataset.location)

    print("\nDataset prepared successfully!")
    print("-" * 50)
    print(f"Path: {os.path.abspath(target_dir)}")
    print(f"Train images: {len(os.listdir(os.path.join(target_dir, 'train', 'images')))}")
    print(f"Val images:   {len(os.listdir(os.path.join(target_dir, 'valid', 'images')) if os.path.exists(os.path.join(target_dir, 'valid', 'images')) else 0)}")
    print("-" * 50)
    print("Mẹo: Hãy kiểm tra file configs/traffic.yaml để đảm bảo đường dẫn 'path' trùng khớp.")

if __name__ == "__main__":
    main()
