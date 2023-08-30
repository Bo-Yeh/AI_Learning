import os
import random

"把資料切割成訓練集/資料集 開始"
img_directory = 'PetImages/'
validation_split = 0.1
subclasses = os.listdir(img_directory)
for subclass in subclasses:
    os.makedirs(img_directory + 'train/' + subclass, exist_ok=True)
    os.makedirs(img_directory + 'validation/' + subclass, exist_ok=True)
    for img in os.listdir(img_directory + subclass):
        rand = random.choices(['train/', 'validation/'], [1-validation_split, validation_split])[0]
        os.rename(img_directory + subclass + '/' + img, img_directory + rand + subclass + '/' + img)
    os.rmdir(img_directory + subclass)
"把資料切割成訓練集/資料集 結束"