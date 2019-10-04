import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import time
import cv2
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pmap", type=str, required=True,
                help="pred_map txt file")
ap.add_argument("-t", "--tmap", type=str, required=True,
                help="true_map txt file")
ap.add_argument("-f", "--fpath", type=str, required=True,
                help="flightpath txt file")
ap.add_argument("-n", "--ntruth", type=str, required=True,
                help="neighborzone_truth txt file")
args = vars(ap.parse_args())


# the size of the whole field, dont change it if you dont change to another dataset(i.e. John's ds)
WIDTH = 42
HEIGHT = 32

EDGE_LEN = 19  # the size of the kernel size. Right now it's 15 * 15

STEPS = 50      # steps for the whole flight path. The python file will stop after these steps plus 2,
# or until there's no available next step to choose.

modelpath = 'models/'    # the folder containing the NN models

imagename = '164030'  # the name of the field.
imagepath = '164030_all/'   # the folder containing all the zones in this field

refdspath = 'refDS_label.h5'  # the file of the reference  dataset

flightpath = 'flightpath.txt'   # the file containing all the flight path

neighborzone_truth_path = 'neighborzone_truth.txt'  # the ground truth data for all the available next steps
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
        if true_map[i] == 0:
            if pred_map[i] >= 0.5:
                gab += 1
            else:
                gag += 1
        else:
            if pred_map[i] < 0.5:
                bag += 1
            else:
                bab += 1
    sum = bab + gag + gab + bag
    #print(f'bab: {bab}, gag: {gag}, gab: {gab}, bag: {bag}, sum: {sum}, all: {l}')
    acc = str(round((bab + gag) * 100 / sum, 4))
    if bab + bag == 0:
        recall_b = 0
    else:
        recall_b = str(round(bab * 100 / (bab + bag), 4))
    if bab + gab == 0:
        precision_b = 0
    else:
        precision_b = str(round(bab * 100 / (bab + gab), 4))
    if gag + gab == 0:
        recall_g = 0
    else:
        recall_g = str(round(gag * 100 / (gag + gab), 4))
    if gag + bag == 0:
        precision_g = 0
    else:
        precision_g = str(round(gag * 100 / (gag + bag), 4))
    filename = 'accuracy.txt'
    f = open(filename, 'a+')
    f.write(f'Accuracy:  {acc}%\n')
    f.write(f'Class:  Precision:  Recall:\n')
    f.write(f'Good      {precision_g}%    {recall_g}%\n')
    f.write(f'Bad       {precision_b}%    {recall_b}%\n')
    f.close()
    #print(f'Accuracy:  {count, acc}%')


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
def get_kernelmap(zone_idx):
    kernel_map = []
    raw_kernelmap = []
    if EDGE_LEN % 2 != 1:
        print('Length of the kernel size must be odd.')
        exit()
    w = zone_idx % WIDTH
    h = zone_idx // WIDTH
    sp_w = w - EDGE_LEN // 2
    sp_h = h - EDGE_LEN // 2
    ep_w = w + EDGE_LEN // 2
    ep_h = h + EDGE_LEN // 2
    while sp_h <= ep_h:
        temp = np.arange(sp_h * WIDTH + sp_w, sp_h * WIDTH + ep_w + 1)
        raw_kernelmap.append(list(temp))
        sp_h += 1
    sp_h = h - EDGE_LEN // 2
    if sp_w < 0:
        sp_w = 0
    if sp_h < 0:
        sp_h = 0
    if ep_w >= WIDTH:
        ep_w = WIDTH - 1
    if ep_h >= HEIGHT:
        ep_h = HEIGHT - 1
    while sp_h <= ep_h:
        temp = np.arange(sp_h * WIDTH + sp_w, sp_h * WIDTH + ep_w + 1)
        kernel_map.append(list(temp))
        sp_h += 1
    kernel_map = np.array(kernel_map)
    raw_kernelmap = np.array(raw_kernelmap)
    return kernel_map, raw_kernelmap

def load_models():
    model_dir = os.listdir(modelpath)
    model_dir.sort()
    models = {}
    for i in range(len(model_dir)):
        models[i + 1] = load_model(modelpath + model_dir[i])
    return models


def load_refdataset():
    h5f = h5py.File(refdspath, 'r')
    ds = h5f['refDS'][:]
    return ds


def init_y_kernelmap(kernelmap):
    h, w = np.shape(kernelmap)
    y_kernelmap = np.zeros((h * w, 9))
    for i in range(h):
        for j in range(w):
            idx = kernelmap[i, j]
            if y_map[idx][4] != 0:
                y_kernelmap[i * w + j] = y_map[idx]
                # print(y_map[idx])
                # print(y_kernelmap[i * w + j])
    return y_kernelmap


def build_y_kernelmap(kernelmap):
    visible_zones = []
    empty_zones = []
    h, w = np.shape(kernelmap)
    y_kernelmap = init_y_kernelmap(kernelmap)
    for i in range(h):
        for j in range(w):
            idx = kernelmap[i, j]
            if y_map[idx][4] != 0:
                visible_zones.append(idx)
            else:
                empty_zones.append(idx)
    # f = open(flightpath, 'a+')
    # for i in visible_zones:
    #     f.write(str(i) + ',')
    # f.write('\n')
    # f.close()
    vis = []
    for zone in visible_zones:
        vis.append(zone)
    search_zones = init_searchzones(empty_zones, visible_zones)
    while len(search_zones) != 0:
        max_value, max_zone, max_neighbor = find_maxzone(search_zones, visible_zones)
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
    return y_kernelmap, vis


def normalize_y_kernelmap(y_kernelmap, kernelmap, raw_kernelmap):
    pred_kernelmap = y_kernelmap[:, 4]
    normalized_y_kernelmap = []
    h, w = np.shape(kernelmap)
    hr, wr = np.shape(raw_kernelmap)
    for i in range(hr):
        for j in range(wr):
            if raw_kernelmap[i][j] not in kernelmap:
                normalized_y_kernelmap.append(-1)
            else:
                row, col = np.where(kernelmap == raw_kernelmap[i][j])
                idx = int(row) * w + int(col)
                normalized_y_kernelmap.append(pred_kernelmap[idx])
    return normalized_y_kernelmap


def get_t_kernelmap(kernelmap):
    row, col = kernelmap.shape
    t_kernelmap = []
    for i in range(row):
        for j in range(col):
            h = kernelmap[i, j] // WIDTH
            w = kernelmap[i, j] % WIDTH
            for img in os.listdir(imagepath):
                name = img.split('_')
                if (int(name[1]) == w) and (int(name[2]) == h):
                    t_kernelmap.append(int(img[-5]))
    return t_kernelmap


def normalize_t_kernelmap(t_kernelmap, kernelmap, raw_kernelmap):
    normalized_t_kernelmap = []
    h, w = np.shape(kernelmap)
    hr, wr = np.shape(raw_kernelmap)
    for i in range(hr):
        for j in range(wr):
            if raw_kernelmap[i][j] not in kernelmap:
                normalized_t_kernelmap.append(-1)
            else:
                row, col = np.where(kernelmap == raw_kernelmap[i][j])
                idx = int(row) * w + int(col)
                normalized_t_kernelmap.append(t_kernelmap[idx])
    return normalized_t_kernelmap


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


def find_maxzone(search_zones, visible_zones):
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


def get_startp():
    sp1 = np.random.randint(0, 4)
    all_neigh = []
    if sp1 == 0:
        w = np.random.randint(0, WIDTH)
        h = sp1
    elif sp1 == 2:
        w = np.random.randint(0, WIDTH)
        h = HEIGHT - 1
    elif sp1 == 1:
        h = np.random.randint(0, HEIGHT)
        w = WIDTH - 1
    else:
        h = np.random.randint(0, HEIGHT)
        w = 0
    startp = h * WIDTH + w
    prename = imagename + '_' + str(w) + '_' + str(h)
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
                pred = models[key].predict(img)[0][1]
                label[key - 1] = pred
            idx = h * WIDTH + w
            y_map[idx] = label
    # print(h * WIDTH + w)
    flightzones.append(startp)
    option = [h * WIDTH + w - 1, h * WIDTH + w + 1, h * WIDTH + w - WIDTH, h * WIDTH + w + WIDTH]
    neighbors = []
    if w == 0:
        if h == 0:
            neighbors.append(h * WIDTH + w + 1)
            neighbors.append(h * WIDTH + w + WIDTH)
        elif h == HEIGHT - 1:
            neighbors.append(h * WIDTH + w + 1)
            neighbors.append(h * WIDTH + w - WIDTH)
        else:
            neighbors.append(h * WIDTH + w + 1)
            neighbors.append(h * WIDTH + w - WIDTH)
            neighbors.append(h * WIDTH + w + WIDTH)
    elif w == WIDTH - 1:
        if h == 0:
            neighbors.append(h * WIDTH + w - 1)
            neighbors.append(h * WIDTH + w + WIDTH)
        elif h == HEIGHT - 1:
            neighbors.append(h * WIDTH + w - 1)
            neighbors.append(h * WIDTH + w - WIDTH)
        else:
            neighbors.append(h * WIDTH + w - 1)
            neighbors.append(h * WIDTH + w - WIDTH)
            neighbors.append(h * WIDTH + w + WIDTH)
    else:
        if h == 0:
            neighbors.append(h * WIDTH + w - 1)
            neighbors.append(h * WIDTH + w + 1)
            neighbors.append(h * WIDTH + w + WIDTH)
        elif h == HEIGHT - 1:
            neighbors.append(h * WIDTH + w - 1)
            neighbors.append(h * WIDTH + w + 1)
            neighbors.append(h * WIDTH + w - WIDTH)
        else:
            pass
    all_neigh.append(startp)

    for i in range(len(option)):
            if option[i] not in neighbors:
                all_neigh.append(option[i])
                all_neigh.append(-1)
            else:
                all_neigh.append(option[i])
                h = option[i] // WIDTH
                w = option[i] % WIDTH
                for zone in os.listdir(imagepath):
                    name = zone.split('_')
                    if (int(name[1]) == w) and (int(name[2]) == h):
                        all_neigh.append(int(zone[-5]))

    return startp, all_neigh


def avail_nextstep(zone_idx, kernelmap):
    w = zone_idx % WIDTH
    h = zone_idx // WIDTH
    neighbors = []
    delzones = []
    if w == 0:
        if h == 0:
            neighbors.append(zone_idx + 1)
            neighbors.append(zone_idx + WIDTH)
        elif h == HEIGHT - 1:
            neighbors.append(zone_idx + 1)
            neighbors.append(zone_idx - WIDTH)
        else:
            neighbors.append(zone_idx + 1)
            neighbors.append(zone_idx - WIDTH)
            neighbors.append(zone_idx + WIDTH)
    elif w == WIDTH - 1:
        if h == 0:
            neighbors.append(zone_idx - 1)
            neighbors.append(zone_idx + WIDTH)
        elif h == HEIGHT - 1:
            neighbors.append(zone_idx - 1)
            neighbors.append(zone_idx - WIDTH)
        else:
            neighbors.append(zone_idx - 1)
            neighbors.append(zone_idx - WIDTH)
            neighbors.append(zone_idx + WIDTH)
    else:
        if h == 0:
            neighbors.append(zone_idx - 1)
            neighbors.append(zone_idx + 1)
            neighbors.append(zone_idx + WIDTH)
        elif h == HEIGHT - 1:
            neighbors.append(zone_idx - 1)
            neighbors.append(zone_idx + 1)
            neighbors.append(zone_idx - WIDTH)
        else:
            neighbors.append(zone_idx - 1)
            neighbors.append(zone_idx + 1)
            neighbors.append(zone_idx - WIDTH)
            neighbors.append(zone_idx + WIDTH)
    for i in neighbors:
        if (i not in kernelmap) or (i in flightzones) or (i >= HEIGHT * WIDTH):
            delzones.append(i)
    for i in delzones:
        neighbors.remove(i)
    return neighbors


def get_nextstep(zone_idx, kernelmap):
    avails = avail_nextstep(zone_idx, kernelmap)
    # print(avails)
    if len(avails) == 0:
        nextstep = -1
    elif len(avails) == 1:
        nextstep = avails[0]
        avails.remove(nextstep)
    else:
        r = np.random.randint(0, len(avails))
        nextstep = avails[r]
        avails.remove(nextstep)
    h = nextstep // WIDTH
    w = nextstep % WIDTH
    for zone in os.listdir(imagepath):
        name = zone.split('_')
        if (int(name[1]) == w) and (int(name[2]) == h):
            img = cv2.imread(imagepath + zone)
            img = cv2.resize(img, (108, 108))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float') / 255.0
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            label = np.zeros(9)
            for i in range(9):
                pred = models[i + 1].predict(img)[0][1]
                label[i] = pred
            y_map[nextstep] = label
            # print(y_map[nextstep][4])
    all_neighbors = get_neighbors_truth(zone_idx, avails)
    flightzones.append(nextstep)
    return nextstep, all_neighbors


def get_neighbors_truth(zone_idx, neighbors):
    all_neigh = []
    all_neigh.append(zone_idx)
    option = [zone_idx - 1, zone_idx + 1, zone_idx - WIDTH, zone_idx + WIDTH]
    l = len(neighbors)
    if l == 0:
        all_neigh.append(option[0])
        all_neigh.append(-1)
        all_neigh.append(option[1])
        all_neigh.append(-1)
        all_neigh.append(option[2])
        all_neigh.append(-1)
        all_neigh.append(option[3])
        all_neigh.append(-1)
    else:
        for i in range(len(option)):
            if option[i] not in neighbors:
                all_neigh.append(option[i])
                all_neigh.append(-1)
            else:
                all_neigh.append(option[i])
                h = option[i] // WIDTH
                w = option[i] % WIDTH
                for zone in os.listdir(imagepath):
                    name = zone.split('_')
                    if (int(name[1]) == w) and (int(name[2]) == h):
                        all_neigh.append(int(zone[-5]))

        '''
        for idx in range(l):
            all_neigh.append(neighbors[idx])
            h = idx // WIDTH
            w = idx % WIDTH
            for zone in os.listdir(imagepath):
                name = zone.split('_')
                if (int(name[1]) == w) and (int(name[2]) == h):
                    all_neigh.append(int(zone[-5]))
        for i in option:
            if i not in neighbors:
                all_neigh.append(i)
                all_neigh.append(-1)
        '''
    return all_neigh


def is_edge(zone_idx):
    w = zone_idx % WIDTH
    h = zone_idx // WIDTH
    if (w in [0, WIDTH - 1]) or (h in [0, HEIGHT - 1]):
        return True
    return False


# Load the models and the reference dataset file
ds = load_refdataset()
models = load_models()
pred_map = []
true_map = []
neighbors_truth = []
flightpaths = []
for loop in range(0, 2):
    # try:
    print(loop)

    count = 0
    flightzones = []

    # Initialize a empty yield map of the whole field
    y_map = init_ymap(WIDTH, HEIGHT)

    # Choose a start point and store the predictions in the y_map
    start_point, neigh_true = get_startp()
    neighbors_truth.append(neigh_true)

    # Generate a kernelmap based on the start point
    kernelmap, raw_kernelmap = get_kernelmap(start_point)

    # Build a yield kernelmap
    y_kernelmap, visible_zones = build_y_kernelmap(kernelmap)
    flightpaths.append(visible_zones)
    normalized_y_kernelmap = normalize_y_kernelmap(y_kernelmap, kernelmap, raw_kernelmap)

    t_kernelmap = get_t_kernelmap(kernelmap)
    normalized_t_kernelmap = normalize_t_kernelmap(t_kernelmap, kernelmap, raw_kernelmap)

    print(np.shape(normalized_y_kernelmap), np.shape(normalized_t_kernelmap))

    pred_map.append(normalized_y_kernelmap)
    true_map.append(normalized_t_kernelmap)
    # show_acc(list(y_kernelmap[:, 4]), t_kernelmap)

    # Find a random next step to go and go into this path finding while loop
    nextstep, neigh_true = get_nextstep(start_point, kernelmap)
    neighbors_truth.append(neigh_true)

    kernelmap, raw_kernelmap = get_kernelmap(nextstep)

    y_kernelmap, visible_zones = build_y_kernelmap(kernelmap)
    flightpaths.append(visible_zones)
    normalized_y_kernelmap = normalize_y_kernelmap(y_kernelmap, kernelmap, raw_kernelmap)

    t_kernelmap = get_t_kernelmap(kernelmap)
    normalized_t_kernelmap = normalize_t_kernelmap(t_kernelmap, kernelmap, raw_kernelmap)

    print(np.shape(normalized_y_kernelmap), np.shape(normalized_t_kernelmap))

    pred_map.append(normalized_y_kernelmap)
    true_map.append(normalized_t_kernelmap)
    # show_acc(list(y_kernelmap[:, 4]), t_kernelmap)
    # while not (is_edge(nextstep, HEIGHT, WIDTH)):
    while count <= STEPS and len(avail_nextstep(nextstep, kernelmap)) != 0:
        nextstep, neigh_true = get_nextstep(nextstep, kernelmap)
        neighbors_truth.append(neigh_true)
        print(neigh_true)
        kernelmap, raw_kernelmap = get_kernelmap(nextstep)

        y_kernelmap, visible_zones = build_y_kernelmap(kernelmap)
        flightpaths.append(visible_zones)
        normalized_y_kernelmap = normalize_y_kernelmap(y_kernelmap, kernelmap, raw_kernelmap)

        t_kernelmap = get_t_kernelmap(kernelmap)
        normalized_t_kernelmap = normalize_t_kernelmap(t_kernelmap, kernelmap, raw_kernelmap)

        print(np.shape(normalized_y_kernelmap), np.shape(normalized_t_kernelmap))

        pred_map.append(normalized_y_kernelmap)
        true_map.append(normalized_t_kernelmap)
        # show_acc(list(y_kernelmap[:, 4]), t_kernelmap)
        count += 1

        # f = open('pred_map.txt', 'a+')
        # for i in range(len(pred_map)):
        #     for j in range(len(pred_map[i])):
        #         f.write(str(pred_map[i][j]) + ',')
        #     f.write('\n')
        # f.close()
        # f = open('true_map.txt', 'a+')
        # for i in range(len(true_map)):
        #     for j in range(len(true_map[i])):
        #         f.write(str(true_map[i][j]) + ',')
        #     f.write('\n')
        # f.close()
        # yieldmap = list(y_map)
        # f = open('yield_map.txt', 'a+')
        # for i in range(len(yieldmap)):
        #     for j in range(len(yieldmap[i])):
        #         f.write(str(yieldmap[i][j]) + ',')
        #     f.write('\n')
        # f.close()

    # except Exception as e:
    #    print(e)

# print(len(pred_map))
# for i in range(len(pred_map)):
#    print(len(pred_map[i]))

pred_map = np.array(pred_map)
true_map = np.array(true_map)
neighbors_truth = np.array(neighbors_truth)
np.savetxt(args["pmap"], pred_map)
np.savetxt(args["tmap"], true_map)
np.savetxt(args["ntruth"], neighbors_truth)
f = open(args["fpath"], 'a+')
for i in range(len(flightpaths)):
    for j in range(len(flightpaths[i])):
        f.write(str(flightpaths[i][j]) + ',')
    f.write('\n')
f.close()
