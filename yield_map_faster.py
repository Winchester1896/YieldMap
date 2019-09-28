from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import time
import cv2
import os
import h5py

WIDTH = 42
HEIGHT = 32
EDGE_LEN = 15
modelpath = 'models/'
visiblepath = '164030_visible//'
imagename = '164030'
imagepath = '164030_all/'
refdspath = 'refDS_label.h5'
# all_list = os.listdir(all_path)
# cover = len(visible_list)
# coverrate = round(cover * 100 / len(all_list), 4)
# print(f'[INFO] Coverage rate: {coverrate}%.')


def init_ymap(width, height):
    y_map = np.zeros((width * height, 9))
    return y_map


# compare the predicted map we got with the ground truth.
# 1 for good class, and -1 for bad
def show_acc(pred_map, true_map):
    l = len(true_map)
    gag = 0 # predict a good one as good, similar below
    bab = 0
    gab = 0
    bag = 0
    if l != len(pred_map):
        print('ERROR: Maps size dont match.')
        exit()
    for i in range(l):
        if true_map[i] == -1:
            if pred_map[i] == -1:
                bab += 1
            else:
                bag += 1
        else:
            if pred_map[i] == 1:
                gag += 1
            else:
                gab += 1
    sum = bab + gag + gab + bag
    acc = str(round((bab + gag) * 100 / sum, 4))
    recall_b = str(round(bab * 100 / (bab + bag), 4))
    precision_b = str(round(bab * 100 / (bab + gab), 4))
    recall_g = str(round(gag * 100 / (gag + gab), 4))
    precision_g = str(round(gag * 100 / (gag + bag), 4))
    print(f'Accuracy:  {acc}%')
    print(f'Class:  Precision:  Recall:')
    print(f'Good      {precision_g}%    {recall_g}%')
    print(f'Bad       {precision_b}%    {recall_b}%')


def get_truemap(all_path, model, width, height):
    true_map = np.zeros((height, width))
    all_dir = os.listdir(all_path)
    for zone in all_dir:
        name = zone.split('_')
        w = int(name[1])
        h = int(name[2])
        img = cv2.imread(all_path + zone)
        img = cv2.resize(img, (108, 108))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float') / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)[0][1]
        if pred >= 0.5:
            true_map[h, w] = -1
        else:
            true_map[h, w] = 1
    if true_map.any(0):
        print('Number of zones and the map size dont match.')
        exit()
    return true_map


def get_predmap(y_map, width, height):
    pred_map = np.zeros((height, width))
    data = y_map[:, 4]
    for idx in range(len(data)):
        w = idx % height
        h = idx // height
        if data[idx] >= 0.5:
            pred_map[h, w] = -1
        else:
            pred_map[h, w] = 1
    return pred_map


# Size of the kernel be decided by edge_len(must be odd), position in the map be decided by zone_idx.
def get_kernelmap(edge_len, zone_idx, height, width):
    kernel_map = []
    if edge_len % 2 != 1:
        print('Length of the kernel size must be odd.')
        exit()
    w = zone_idx % height
    h = zone_idx // height
    sp_w = w - edge_len // 2
    sp_h = h - edge_len // 2
    ep_w = w + edge_len // 2
    ep_h = h + edge_len // 2
    if sp_w < 0:
        sp_w = 0
    if sp_h < 0:
        sp_h = 0
    if ep_w > width:
        ep_w = width
    if ep_h > height:
        ep_h = height
    while sp_h != ep_h:
        temp = np.arange(sp_h * width + sp_w, sp_h * width + ep_w)
        kernel_map.append(list(temp))
        sp_h += 1
    kernel_map = np.array(kernel_map)
    return kernel_map


def load_models(modelpath):
    model_dir = os.listdir(modelpath)
    model_dir.sort()
    models = {}
    for i in range(len(model_dir)):
        models[i + 1] = load_model(modelpath + model_dir[i])
    return models


def load_refdataset(filepath):
    h5f = h5py.File(filepath, 'r')
    ds = h5f['refDS'][:]
    return ds


def init_y_kernelmap(kernelmap, y_map):
    h, w = np.shape(kernelmap)
    y_kernelmap = np.zeros((h * w, 9))
    for i in kernelmap:
        if y_map[i][4] != 0:
            row, col = np.where(kernelmap == i)
            y_kernelmap[int(row) * w + int(col)] = y_map[i]
    return y_kernelmap


def build_y_kernelmap(kernelmap, y_map, ds, width):
    visible_zones = []
    empty_zones = []
    h, w = np.shape(kernelmap)
    y_kernelmap = init_y_kernelmap(kernelmap, y_map)
    for i in kernelmap:
        if y_map[i][4] != 0:
            visible_zones.append(i)
    for i in kernelmap:
        if i not in visible_zones:
            empty_zones.append(i)
    search_zones = init_searchzones(empty_zones, visible_zones)
    while len(search_zones) != 0:
        max_value, max_zone, max_neighbor = find_maxzone(search_zones, visible_zones, width)
        total_pred = 0.0
        for key in max_neighbor:
            row, col = np.where(kernelmap == max_neighbor[key])
            pred = y_kernelmap[int(row) * w + int(col)][key - 1]
            total_pred += pred
        final_pred = total_pred / max_value
        min_idx = (np.abs(ds[:, 4] - final_pred)).argmin()
        row, col = np.where(kernelmap == max_zone)
        y_kernelmap[int(row) * w + int(col)] = ds[min_idx]
        update_map(max_zone, empty_zones, visible_zones, search_zones)
    return y_kernelmap


def get_t_kernelmap(kernelmap, imagepath, imagename):



'''
ds = load_refdataset('refDS_label.h5')
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

end = time.time() - start
print(len(visible_zones) == (WIDTH * HEIGHT))
print(f'Time spent: {end}s.')
'''


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


def find_maxzone(search_zones, visible_zones, width):
    max_value = 0
    max_zone = ''
    max_neighbor = {}
    for idx in search_zones:
        count = 0
        closest = {}
        for zone in visible_zones:
            if zone == idx - width - 1:
                count += 1
                closest[1] = zone
            elif zone == idx - width:
                count += 1
                closest[2] = zone
            elif zone == idx - width + 1:
                count += 1
                closest[3] = zone
            elif zone == idx - 1:
                count += 1
                closest[4] = zone
            elif zone == idx + 1:
                count += 1
                closest[6] = zone
            elif zone == idx + width - 1:
                count += 1
                closest[7] = zone
            elif zone == idx + width:
                count += 1
                closest[8] = zone
            elif zone == idx + width + 1:
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


def get_startp(height, width, y_map, models, imagepath):
    sp1 = np.random.randint(0, 4)
    if sp1 == 0:
        w = np.random.randint(0, width)
        h = sp1
    elif sp1 == 2:
        w = np.random.randint(0, width)
        h = height - 1
    elif sp1 == 1:
        h = np.random.randint(0, height)
        w = width - 1
    else:
        h = np.random.randint(0, height)
        w = 0
    prename = imagename + '_' + w + '_' + h
    for zone in os.listdir(imagepath):
        if (zone == prename + '_0.jpg') or (zone == prename + '_1.jpg'):
            img = cv2.imread(imagepath + zone)
            img = cv2.resize(img, (108, 108))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float') / 255.0
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            label = np.zeros(9)
            for key in models:
                pred = models[key].predict(img)[0][key]
                label[key - 1] = pred
            idx = h * WIDTH + w
            y_map[idx] = label
    return h * width + w


def avail_nextstep(zone_idx, kernelmap, width, y_map):
    neighbors = []
    neighbors.append(zone_idx - 1)
    neighbors.append(zone_idx + 1)
    neighbors.append(zone_idx - width)
    neighbors.append(zone_idx + width)
    for i in neighbors:
        if (i in kernelmap) and (y_map[i][4] != 0):
            pass
        else:
            neighbors.remove(i)
    return neighbors


def get_nextstep(zone_idx, kernelmap, width, y_map):
    avails = avail_nextstep(zone_idx, kernelmap, width, y_map)
    r = np.random.randint(0, len(avails))
    nextstep = avails[r]
    return nextstep


def is_edge(zone_idx, HEIGHT, WIDTH):
    w = zone_idx % HEIGHT
    h = zone_idx // HEIGHT
    if (w in [0, WIDTH - 1]) or (h in [0, HEIGHT - 1]):
        return True
    return False


# Load the models and the reference dataset file
ds = load_refdataset(refdspath)
models = load_models(modelpath)

# Initialize a empty yield map of the whole field
y_map = init_ymap(WIDTH, HEIGHT)

# Choose a start point and store the predictions in the y_map
start_point = get_startp(HEIGHT, WIDTH, y_map, models, imagepath)

# Generate a kernelmap based on the start point
kernelmap = get_kernelmap(EDGE_LEN, start_point, HEIGHT, WIDTH)

# Build a yield kernelmap
y_kernelmap = build_y_kernelmap(kernelmap, y_map, ds, WIDTH)
t_kernelmap = get_t_kernelmap(kernelmap, imagepath, imagename)
pred_map = []
true_map = []
pred_map.append(list(y_kernelmap[:, 4]))
# Find a random next step to go and go into this path finding while loop
nextstep = get_nextstep(start_point, kernelmap, WIDTH, y_map)
kernelmap = get_kernelmap(EDGE_LEN, nextstep, HEIGHT, WIDTH)
y_kernelmap = build_y_kernelmap(kernelmap, y_map, ds, WIDTH)
pred_map.append(list(y_kernelmap[:, 4]))
while


