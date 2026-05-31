import os
import csv
import matplotlib.pyplot as plt

def main():
    train_dir = "runs/train"
    
    if not os.path.exists(train_dir):
        print(f"[ERROR] Training directory '{train_dir}' not found. Please run training first.")
        return

    # Find all completed runs containing results.csv
    runs = []
    for run_name in os.listdir(train_dir):
        run_path = os.path.join(train_dir, run_name)
        csv_path = os.path.join(run_path, "results.csv")
        if os.path.isdir(run_path) and os.path.exists(csv_path):
            runs.append((run_name, csv_path))

    if not runs:
        print(f"[ERROR] No training runs with 'results.csv' found in '{train_dir}'.")
        return

    print(f"Found {len(runs)} training runs to plot:")
    for run_name, _ in runs:
        print(f" - {run_name}")

    # Set up matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Create figure with 2 subplots (Loss and mAP50)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for run_name, csv_path in runs:
        # Load CSV data using standard csv module
        try:
            epochs = []
            train_loss = []
            val_loss = []
            map50 = []
            
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # Strip keys to remove potential whitespace issues from YOLO headers
                reader.fieldnames = [name.strip() for name in reader.fieldnames]
                
                for row in reader:
                    # Clean the keys inside the row dict as well
                    clean_row = {k.strip(): v.strip() for k, v in row.items()}
                    
                    epochs.append(int(clean_row['epoch']))
                    
                    # Box loss + Class loss
                    t_loss = float(clean_row['train/box_loss']) + float(clean_row['train/cls_loss'])
                    v_loss = float(clean_row['val/box_loss']) + float(clean_row['val/cls_loss'])
                    
                    train_loss.append(t_loss)
                    val_loss.append(v_loss)
                    
                    map50.append(float(clean_row['metrics/mAP50(B)']))
            
            # Plot train/val losses
            line1, = ax1.plot(epochs, train_loss, linestyle='-', label=f"{run_name} (Train)")
            ax1.plot(epochs, val_loss, linestyle='--', color=line1.get_color(), label=f"{run_name} (Val)")
            
            # Plot mAP50
            ax2.plot(epochs, map50, linestyle='-', label=run_name)
            
        except Exception as e:
            print(f"[WARNING] Could not parse results for '{run_name}': {e}")
            continue

    # Customize Loss Subplot
    ax1.set_title("Training & Validation Loss Comparison", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epochs", fontsize=10)
    ax1.set_ylabel("Total Loss (Box + Cls)", fontsize=10)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Customize mAP50 Subplot
    ax2.set_title("mAP@0.5 Metric Comparison", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epochs", fontsize=10)
    ax2.set_ylabel("mAP@0.5", fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    # Save the output plot
    output_path = os.path.join(train_dir, "comparison_plots.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nSuccess: Comparison plot saved to: {os.path.abspath(output_path)}")
    plt.close()

if __name__ == "__main__":
    main()
