from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import time
import cv2
import os
import h5py

WIDTH = 20
HEIGHT = 20
modelpath = 'models/'
visiblepath = 'smaller_visible/'
# all_path = '164030_all/'
# all_list = os.listdir(all_path)
# cover = len(visible_list)
# coverrate = round(cover * 100 / len(all_list), 4)
# print(f'[INFO] Coverage rate: {coverrate}%.')


def initialization(modelpath, visiblepath, WIDTH, HEIGHT):
    models = os.listdir(modelpath)
    models.sort()
    visible_list = os.listdir(visiblepath)
    visible_list.sort()
    a = np.arange(WIDTH * HEIGHT)
    empty_zones = list(a)
    visible_zones = []
    y_map = np.zeros((WIDTH * HEIGHT, 9))
    model1 = load_model(modelpath + models[0])
    model2 = load_model(modelpath + models[1])
    model3 = load_model(modelpath + models[2])
    model4 = load_model(modelpath + models[3])
    model5 = load_model(modelpath + models[4])
    model6 = load_model(modelpath + models[5])
    model7 = load_model(modelpath + models[6])
    model8 = load_model(modelpath + models[7])
    model9 = load_model(modelpath + models[8])
    for zone in visible_list:
        label = np.zeros(9)
        name = zone.split('_')
        w = int(name[1])
        h = int(name[2])
        img = cv2.imread(visiblepath + zone)
        img = cv2.resize(img, (108, 108))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float') / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        pred = model1.predict(img)[0][1]
        label[0] = pred
        pred = model2.predict(img)[0][1]
        label[1] = pred
        pred = model3.predict(img)[0][1]
        label[2] = pred
        pred = model4.predict(img)[0][1]
        label[3] = pred
        pred = model5.predict(img)[0][1]
        label[4] = pred
        pred = model6.predict(img)[0][1]
        label[5] = pred
        pred = model7.predict(img)[0][1]
        label[6] = pred
        pred = model8.predict(img)[0][1]
        label[7] = pred
        pred = model9.predict(img)[0][1]
        label[8] = pred
        idx = h * WIDTH + w
        y_map[idx] = label
        empty_zones.remove(idx)
        visible_zones.append(idx)
    return y_map, empty_zones, visible_zones


def findzone(search_zones, visible_zones, WIDTH):
    max_value = 0
    max_zone = ''
    max_neighbor = {}
    for idx in search_zones:
        count = 0
        closest = {}
        for zone in visible_zones:
            if zone == idx - WIDTH - 1:
                count += 1
                closest[1] = zone
            elif zone == idx - WIDTH:
                count += 1
                closest[2] = zone
            elif zone == idx - WIDTH + 1:
                count += 1
                closest[3] = zone
            elif zone == idx - 1:
                count += 1
                closest[4] = zone
            elif zone == idx + 1:
                count += 1
                closest[6] = zone
            elif zone == idx + WIDTH - 1:
                count += 1
                closest[7] = zone
            elif zone == idx + WIDTH:
                count += 1
                closest[8] = zone
            elif zone == idx + WIDTH + 1:
                count += 1
                closest[9] = zone
        if count > max_value:
            max_zone = idx
            max_value = count
            max_neighbor = closest
    return max_value, max_zone, max_neighbor


def neighbors(zone):
    neibor = []
    neibor.append(zone - WIDTH - 1)
    neibor.append(zone - WIDTH)
    neibor.append(zone - WIDTH + 1)
    neibor.append(zone - 1)
    neibor.append(zone + 1)
    neibor.append(zone + WIDTH - 1)
    neibor.append(zone + WIDTH)
    neibor.append(zone + WIDTH + 1)
    return neibor


def init_searchzones(empty_zones, visible_zones):
    search_zones = []
    for zone in visible_zones:
        for neibor in neighbors(zone):
            if neibor in empty_zones:
                empty_zones.remove(neibor)
                search_zones.append(neibor)
    return search_zones


def update_map(max_zone, empty_zones, visible_zones, search_zones):
    if len(empty_zones) != 0:
        for neighbor in neighbors(max_zone):
            if neighbor in empty_zones:
                empty_zones.remove(neighbor)
                search_zones.append(neighbor)
    visible_zones.append(max_zone)
    search_zones.remove(max_zone)


h5f = h5py.File('refDS_label.h5', 'r')
ds = h5f['refDS'][:]

y_map, empty_zones, visible_zones = initialization(modelpath, visiblepath, WIDTH, HEIGHT)
search_zones = init_searchzones(empty_zones, visible_zones)
start = time.time()
while len(search_zones) != 0:
    max_value, max_zone, max_neighbor = findzone(search_zones, visible_zones, WIDTH)
    total_pred = 0.0
    for key in max_neighbor:
        pred = y_map[max_neighbor[key]][key - 1]
        total_pred += pred
    final_pred = total_pred / max_value
    min_idx = (np.abs(ds[:, 4] - final_pred)).argmin()
    y_map[max_zone] = ds[min_idx]
    update_map(max_zone, empty_zones, visible_zones, search_zones)
    # print(max_zone)

end = time.time() - start
# visible_zones.sort()
# print(visible_zones)
print(len(visible_zones) == (WIDTH * HEIGHT))
# print(y_map)
print(f'Time spent: {end}s.')



