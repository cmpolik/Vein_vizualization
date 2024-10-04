import os
import supervisely as sly
from supervisely.io.fs import get_file_name, get_file_name_with_ext

def convert_supervisely_to_yolo(sly_project_dir, output_dir):
    if not os.path.exists(os.path.join(sly_project_dir, 'meta.json')):
        raise FileNotFoundError(f"File with path {os.path.join(sly_project_dir, 'meta.json')} was not found.")
    
    project = sly.Project(sly_project_dir, sly.OpenMode.READ)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    # Сопоставление имени класса с его индексом
    class_to_index = {obj_class.name: idx for idx, obj_class in enumerate(project.meta.obj_classes)}

    for dataset in project.datasets:
        dataset_images_dir = os.path.join(output_dir, 'images', dataset.name)
        dataset_labels_dir = os.path.join(output_dir, 'labels', dataset.name)
        os.makedirs(dataset_images_dir, exist_ok=True)
        os.makedirs(dataset_labels_dir, exist_ok=True)

        for item_name in dataset:
            item_path = dataset.get_item_paths(item_name)
            ann = sly.Annotation.load_json_file(item_path.ann_path, project.meta)

            # Save image
            img_dest_path = os.path.join(dataset_images_dir, get_file_name_with_ext(item_path.img_path))
            sly.image.write(img_dest_path, sly.image.read(item_path.img_path))

            # Convert annotation to YOLO format
            img_height, img_width = ann.img_size
            yolo_annotations = []

            for label in ann.labels:
                obj_class = label.obj_class
                bbox = label.geometry.to_bbox()
                x_center = (bbox.left + bbox.width / 2) / img_width
                y_center = (bbox.top + bbox.height / 2) / img_height
                width = bbox.width / img_width
                height = bbox.height / img_height

                # Используем индекс класса из class_to_index
                class_index = class_to_index[obj_class.name]
                yolo_annotations.append(f"{class_index} {x_center} {y_center} {width} {height}\n")

            # Save annotation
            label_dest_path = os.path.join(dataset_labels_dir, f"{get_file_name(item_path.img_path)}.txt")
            with open(label_dest_path, 'w') as f:
                f.writelines(yolo_annotations)

# Пример использования
sly_project_dir = '/home/samvel/Desktop/YOLOVeins/dataset'
output_dir = '/home/samvel/Desktop/YOLOVeins/aa'
convert_supervisely_to_yolo(sly_project_dir, output_dir)

