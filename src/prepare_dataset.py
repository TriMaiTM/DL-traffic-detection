import os
import urllib.request
import zipfile
import shutil
import random

def main():
    zip_url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    zip_path = "data/coco128.zip"
    extract_path = "data/coco128_extracted"
    dataset_path = "data/dataset"

    # Create directories
    os.makedirs("data", exist_ok=True)
    
    # 1. Download coco128.zip
    if not os.path.exists(zip_path):
        print(f"Downloading coco128 dataset from: {zip_url}...")
        urllib.request.urlretrieve(zip_url, zip_path)
        print("Download completed.")
    else:
        print("ZIP file already exists.")

    # 2. Extract coco128.zip
    if not os.path.exists(extract_path):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction completed.")
    else:
        print("Dataset already extracted.")

    # 3. Create target dataset structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dataset_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, split, 'labels'), exist_ok=True)

    # Classes we want to keep and their new mapping
    # COCO Class mapping: 2 -> car, 3 -> motorcycle, 7 -> truck
    coco_to_new = {
        2: 0,  # car
        3: 1,  # motorcycle
        7: 2   # truck
    }

    coco128_labels_dir = os.path.join(extract_path, "coco128", "labels", "train2017")
    coco128_images_dir = os.path.join(extract_path, "coco128", "images", "train2017")

    valid_samples = []

    print("Filtering and remapping labels...")
    for label_file in os.listdir(coco128_labels_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(coco128_labels_dir, label_file)
        new_labels = []

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            # If the class is car, motorcycle, or truck, keep and map it
            if class_id in coco_to_new:
                new_class_id = coco_to_new[class_id]
                new_labels.append(f"{new_class_id} " + " ".join(parts[1:]))

        # Only keep the image if it has at least one target class
        if new_labels:
            image_name = label_file.replace(".txt", ".jpg")
            image_path = os.path.join(coco128_images_dir, image_name)
            
            if os.path.exists(image_path):
                valid_samples.append({
                    'image_path': image_path,
                    'label_content': "\n".join(new_labels),
                    'label_name': label_file
                })

    print(f"Found {len(valid_samples)} images containing target classes.")

    # 4. Split into Train (80%) and Val (20%)
    random.seed(42)  # For reproducibility
    random.shuffle(valid_samples)
    
    split_index = int(len(valid_samples) * 0.8)
    train_samples = valid_samples[:split_index]
    val_samples = valid_samples[split_index:]

    # Helper function to write split
    def write_split(samples, split_name):
        class_counts = {0: 0, 1: 0, 2: 0}
        for sample in samples:
            # Copy image
            dest_img = os.path.join(dataset_path, split_name, 'images', os.path.basename(sample['image_path']))
            shutil.copy(sample['image_path'], dest_img)
            
            # Write label file
            dest_lbl = os.path.join(dataset_path, split_name, 'labels', sample['label_name'])
            with open(dest_lbl, 'w') as f:
                f.write(sample['label_content'])

            # Count classes
            for line in sample['label_content'].split('\n'):
                if line:
                    c_id = int(line.split()[0])
                    class_counts[c_id] += 1
        return class_counts

    print("Writing train split...")
    train_counts = write_split(train_samples, "train")
    print("Writing validation split...")
    val_counts = write_split(val_samples, "val")

    print("\nDataset preparation completed successfully!")
    print("-" * 40)
    print(f"Total Train Images: {len(train_samples)}")
    print(f"Total Val Images:   {len(val_samples)}")
    print("-" * 40)
    print("Class distribution (number of objects):")
    print(f" - Car (0):        Train={train_counts[0]}, Val={val_counts[0]}")
    print(f" - Motorcycle (1): Train={train_counts[1]}, Val={val_counts[1]}")
    print(f" - Truck (2):      Train={train_counts[2]}, Val={val_counts[2]}")
    print("-" * 40)

if __name__ == "__main__":
    main()
