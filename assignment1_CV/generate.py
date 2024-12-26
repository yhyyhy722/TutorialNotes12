import csv
import os

# 设置路径
train_folder = "./flower_dataset/train"         # 替换为你的训练集文件夹路径
output_csv = "./flower_dataset/train.csv"              # 替换为输出CSV文件路径
mapping = {"carnation": 0, "iris": 1, "bellflower": 2, "california_poppy": 3, "rose": 4, 
           "astilbe": 5, "tulip": 6, "calendula": 7, "dandelion": 8, "coreopsis": 9, "black_eyed_susan": 10,
           "water_lily": 11, "sunflower": 12, "common_daisy": 13}  # 类别映射

def generate_csv(folder_path, mapping, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image:FILE", "category"])
        for class_name, class_id in mapping.items():
            class_folder = os.path.join(folder_path, class_name)
            for filename in os.listdir(class_folder):
                if filename.endswith(".jpg"):
                    relative_path = os.path.join("train", class_name, filename)
                    if os.path.isfile(os.path.join(class_folder, filename)):
                        writer.writerow([relative_path, class_id])

if __name__ == "__main__":
    generate_csv(train_folder, mapping, output_csv)