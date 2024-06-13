import cv2
import numpy as np
import os
import random

if False:
    images_village_island_independent = [
        'village_island_independent163.png',
        'village_island_independent23.png',
        'village_island_independent81.png'
    ]

    images_hababam_independent = [
        'hababam27.png',
        'hababam42.png',
        'hababam126.png'
    ]

    # merge them vertically
    village_island_independent = []
    for img in images_village_island_independent:
        village_island_independent.append(cv2.imread(img))

    hababam_independent = []
    for img in images_hababam_independent:
        hababam_independent.append(cv2.imread(img))

    village_island_independent = np.concatenate(village_island_independent, axis=0)
    hababam_independent = np.concatenate(hababam_independent, axis=0)

    # save them
    cv2.imwrite('io_village_island_independent.png', village_island_independent)
    cv2.imwrite('io_hababam_independent.png', hababam_independent)

if False:
    folder_hababam = 'hababam_dataset_independent/test/rgb'
    folder_village_island = 'village_dataset_independent/test/rgb'
    # randomly read 25 images in folder
    images_hababam = []
    images_village_island = []
    images_hababam = os.listdir(folder_hababam)
    images_village_island = os.listdir(folder_village_island)
    random.shuffle(images_hababam)
    random.shuffle(images_village_island)
    images_hababam = images_hababam[:25]
    images_village_island = images_village_island[:25]
    # merge them vertically and horizontally with some padding
    hababam = []
    for img in images_hababam:
        image = cv2.imread(os.path.join(folder_hababam, img))
        # add padding to make the frame square
        h, w, _ = image.shape
        padding = (w - h) // 2
        image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        image = cv2.resize(image, (256, 256))
        # add 5 pixel padding to all sides
        image = cv2.copyMakeBorder(image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        hababam.append(image)
    village_island = []
    for img in images_village_island:
        image = cv2.imread(os.path.join(folder_village_island, img))
        # add padding to make the frame square
        h, w, _ = image.shape
        padding = (w - h) // 2
        image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        image = cv2.resize(image, (256, 256))
        # add 5 pixel padding to all sides
        image = cv2.copyMakeBorder(image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        village_island.append(image)
    # make them 5x5
    hababam_rows = []
    for i in range(5):
        hababam_rows.append(np.concatenate(hababam[i*5:(i+1)*5], axis=1))
    hababam = np.concatenate(hababam_rows, axis=0)
    village_island_rows = []
    for i in range(5):
        village_island_rows.append(np.concatenate(village_island[i*5:(i+1)*5], axis=1))
    village_island = np.concatenate(village_island_rows, axis=0)
    # save them
    cv2.imwrite('io_raw_hababam.png', hababam)
    cv2.imwrite('io_raw_village_island.png', village_island)