import os
import random
import shutil

def split_dataset(dataset_dir, train_subdir='train', val_subdir='val', val_split=0.12):
    # Paths to train and val directories
    train_images_dir = os.path.join(dataset_dir, 'images', train_subdir)
    val_images_dir = os.path.join(dataset_dir, 'images', val_subdir)
    train_labels_dir = os.path.join(dataset_dir, 'labels', train_subdir)
    val_labels_dir = os.path.join(dataset_dir, 'labels', val_subdir)

    # Create val directories if they don't exist
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # List all image files in train directory
    image_files = [f for f in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, f))]
    num_val = int(len(image_files) * val_split)

    # Randomly select files for val set
    val_files = random.sample(image_files, num_val)

    for file_name in val_files:
        # Move image file
        shutil.move(os.path.join(train_images_dir, file_name), os.path.join(val_images_dir, file_name))

        # Move corresponding label file
        label_name = os.path.splitext(file_name)[0] + '.txt'
        shutil.move(os.path.join(train_labels_dir, label_name), os.path.join(val_labels_dir, label_name))

    print(f"Moved {num_val} files to validation set.")

# Usage
dataset_directory = '/home/samvel/Desktop/YOLOVeins/DATA'
split_dataset(dataset_directory)

