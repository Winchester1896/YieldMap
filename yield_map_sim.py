from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import time
import cv2
import os
import h5py
from matplotlib import pyplot as plt
import csv
from sklearn.neighbors import NearestNeighbors

ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--pmap", type=str, required=True,
#                help="pred_map txt file")
#ap.add_argument("-t", "--tmap", type=str, required=True,
#                help="true_map txt file")
#ap.add_argument("-f", "--fpath", type=str, required=True,
#                help="flightpath txt file")
#ap.add_argument("-n", "--ntruth", type=str, required=True,
#                help="neighborzone_truth txt file")
ap.add_argument("-knn", "--knndataset", type=str, required=True,
                help="KNN dataset")
args = vars(ap.parse_args())


# the size 0f the whole field, dont change it if you dont change to another dataset(i.e. John's ds)
WIDTH = 50
HEIGHT = 33

EDGE_LEN = 15  # the size of the kernel size. Right now it's 15 * 15

STEPS = 50   # steps for the whole flight path. The python file will stop after these steps plus 2,
# or until there's no available next step to choose.

modelpath = 'models/'    # the folder containing the NN models

#imagename = '164030'  # the name of the field.
#imagepath = '164030_all/'   # the folder containing all the zones in this field

refdspath = 'refDS_label.h5'  # the file of the reference  dataset

flightpath = 'flightpath.txt'   # the file containing all the flight path

neighborzone_truth_path = 'neighborzone_truth.txt'  # the ground truth data for all the available next steps
# all_list = os.listdir(all_path)
# cover = len(visible_list)
# coverrate = round(cover * 100 / len(all_list), 4)
# print(f'[INFO] Coverage rate: {coverrate}%.')

testImg = 'test.JPG'

def readKNN():
    #ds = []
    #with open(args['knndataset'], 'rt', encoding='utf-8') as data:
    #    reader = csv.reader(data)
    #    for row in reader:
    #        #row = [float(i) for i in row]
    #        ds.append(row)

    return np.genfromtxt(args['knndataset'], delimiter=',')

def findGEx(im):
    imWidth, imHeight, chan = im.shape

    GI = 0
    total = 0
    speed = 8

    for i in range(0, int(imWidth/speed)):
        for j in range(0, int(imHeight/speed)):
            R,G,B = im[i*speed,j*speed]
            GI += 2*G - R - B
            total = total+1

    if(float(GI/total) < 35):
        return 1
    else:
        return 0

def load_map():
    img = cv2.imread(testImg)
    shp = img.shape
    height = int(shp[0]/108)
    width = int(shp[1]/108)
    GExMap = np.zeros((width, height))
    IMGMap = np.zeros((width, height, 108, 108, 3))

    for i in range(width):
        for j in range(height):
            crop = img[j*108:(j+1)*108, i*108:(i+1)*108]
            GEx = findGEx(crop)
            GExMap[i,j] = GEx
            IMGMap[i,j] = crop

    plt.imshow(GExMap)

    return GExMap, IMGMap

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
    print(f'bab: {bab}, gag: {gag}, gab: {gab}, bag: {bag}, sum: {sum}, all: {l}')
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
    print(f'Accuracy:  {count, acc}%')

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


def get_wholefield():
    fieldmap = []
    sp_w = 0
    sp_h = 0
    ep_w = WIDTH - 1
    ep_h = HEIGHT - 1
    while sp_h <= ep_h:
        temp = np.arange(sp_h * WIDTH + sp_w, sp_h * WIDTH + ep_w + 1)
        fieldmap.append(list(temp))
        sp_h += 1
    fieldmap = np.array(fieldmap)
    return fieldmap


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
        ep_w = WIDTH-1
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
    f = open(flightpath, 'a+')
    for i in visible_zones:
        f.write(str(i) + ',')
    f.write('\n')
    f.close()
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
    return y_kernelmap


def get_t_kernelmap(kernelmap, IMGMap):
    row, col = kernelmap.shape
    t_kernelmap = []
    for i in range(row):
        for j in range(col):
            h = kernelmap[i, j] // WIDTH
            w = kernelmap[i, j] % WIDTH
            t_kernelmap.append(IMGMap[w][h])
            #for img in os.listdir(imagepath):
            #    name = img.split('_')
            #    if (int(name[1]) == w) and (int(name[2]) == h):
            #        t_kernelmap.append(int(img[-5]))
    return t_kernelmap

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


def get_startp(IMGMap):
    sp1 = np.random.randint(0, 4)
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

    img = IMGMap[w][h]
    img = img.astype('float')/255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    label = np.zeros(9)
    for key in models:
        pred = models[key].predict(img)[0][1]
        label[key-1] = pred
    idx = h*WIDTH+w
    y_map[idx] = label
    flightzones.append(h*WIDTH+w)

    print(label)

    return h*WIDTH+w

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


def get_nextstep(zone_idx, kernelmap, kernel_t, kernel_y, IMGMap, GTMap, knn, y_map):
    avails = avail_nextstep(zone_idx, kernelmap)
    print(avails)

    print(len(kernel_t))

    query = []
    for val in kernel_t:
        query.append(val)

    for val in kernel_y:
        query.append(val)

    nbors = NearestNeighbors(9)
    nbors.fit(knn[:,0:-4])
    result = nbors.kneighbors([query])
    neighbors = result[1][0]
    preds = [[0,0],[0,0],[0,0],[0,0]]
    for n in neighbors:
        pred = knn[n][-4:]
        for d in range(0,4):
            if(pred[d] > -1):
                preds[d][0] += 1
                preds[d][1] += pred[d]

    gain = [p[1] / (p[0]-0.00001) for p in preds]

    zones = [zone_idx-1, zone_idx+1, zone_idx-WIDTH, zone_idx+WIDTH]

    maxErr = -1
    for i in range(len(zones)):
        if zones[i] in avails:
            idx = zones[i]
            r = np.mean(y_map[idx])
            pred_err = abs(r-gain[i])
            if(pred_err > maxErr):
                maxErr = pred_err
                nextstep = idx

    h = nextstep // WIDTH
    w = nextstep % WIDTH

    img = IMGMap[w-1][h-1]
    img = img.astype('float')/255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    label = np.zeros(9)
    for i in range(9):
        pred = models[i+1].predict(img)[0][1]
        label[i] = pred
    y_map[nextstep] = label
    print(y_map[nextstep][4])

    all_neighbors = get_neighbors_truth(zone_idx, avails, GTMap)
    flightzones.append(nextstep)
    return nextstep

def get_neighbors_truth(zone_idx, neighbors, GTMap):
    f = open(neighborzone_truth_path, 'a+')
    if len(neighbors) == 0:
        f.write(str(zone_idx) + ': 0\n')
    else:
        f.write(str(zone_idx) + ': ' + str(len(neighbors)))
        print("LEN: "+str(len(neighbors)))
        for idx in neighbors:
            h = idx // WIDTH
            w = idx % WIDTH
            #for zone in os.listdir(imagepath):
            #    name = zone.split('_')
            #    if (int(name[1]) == w) and (int(name[2]) == h):
            f.write(', ' + str(idx) + ': ' + str(GTMap[w-1][h-1]))
        f.write('\n')
    f.close()


def is_edge(zone_idx):
    w = zone_idx % WIDTH
    h = zone_idx // WIDTH
    if (w in [0, WIDTH - 1]) or (h in [0, HEIGHT - 1]):
        return True
    return False

knn = readKNN()

GTMap, IMGMap = load_map()

ds = load_refdataset()
models = load_models()

for i in range(1):
        # Load the models and the reference dataset file
        count = 0
        flightzones = []

        # Initialize a empty yield map of the whole field
        y_map = init_ymap(WIDTH, HEIGHT)

        # Choose a start point and store the predictions in the y_map
        start_point = get_startp(IMGMap)

        #print(y_map[start_point])
        # Generate a kernelmap based on the start point
        kernelmap, raw_kernelmap = get_kernelmap(start_point)

        # Build a yield kernelmap
        y_kernelmap = build_y_kernelmap(kernelmap)
        normalized_y_kernelmap = normalize_y_kernelmap(y_kernelmap, kernelmap, raw_kernelmap)

        t_kernelmap = get_t_kernelmap(kernelmap, GTMap)
        normalized_t_kernelmap = normalize_t_kernelmap(t_kernelmap, kernelmap, raw_kernelmap)

        pred_map = []
        true_map = []
        pred_map.append(list(y_kernelmap[:, 4]))
        true_map.append(t_kernelmap)

        show_acc(list(y_kernelmap[:, 4]), t_kernelmap)
        # Find a random next step to go and go into this path finding while loop
        nextstep = get_nextstep(start_point, kernelmap, normalized_t_kernelmap, normalized_y_kernelmap, IMGMap, GTMap, knn, y_map)
        #print(nextstep)
        #print(y_map[nextstep][4])
        kernelmap, raw_kernelmap = get_kernelmap(nextstep)
        y_kernelmap = build_y_kernelmap(kernelmap)
        normalized_y_kernelmap = normalize_y_kernelmap(y_kernelmap, kernelmap, raw_kernelmap)

        t_kernelmap = get_t_kernelmap(kernelmap, GTMap)
        normalized_t_kernelmap = normalize_t_kernelmap(t_kernelmap, kernelmap, raw_kernelmap)

        pred_map.append(list(y_kernelmap[:, 4]))
        true_map.append(t_kernelmap)
        show_acc(list(y_kernelmap[:, 4]), t_kernelmap)
        # while not (is_edge(nextstep, HEIGHT, WIDTH)):
        while count <= STEPS and len(avail_nextstep(nextstep, kernelmap)) != 0:
            nextstep = get_nextstep(nextstep, kernelmap, normalized_t_kernelmap, normalized_y_kernelmap, IMGMap, GTMap, knn, y_map)
            #print(nextstep)
            #print(y_map[nextstep][4])
            kernelmap, raw_kernelmap = get_kernelmap(nextstep)
            y_kernelmap = build_y_kernelmap(kernelmap)
            normalized_y_kernelmap = normalize_y_kernelmap(y_kernelmap, kernelmap, raw_kernelmap)

            t_kernelmap = get_t_kernelmap(kernelmap, GTMap)
            normalized_t_kernelmap = normalize_t_kernelmap(t_kernelmap, kernelmap, raw_kernelmap)

            pred_map.append(list(y_kernelmap[:, 4]))
            true_map.append(t_kernelmap)
            show_acc(list(y_kernelmap[:, 4]), t_kernelmap)
            count += 1

        wholefield = get_wholefield()
        y_fieldmap = build_y_kernelmap(wholefield)
        show_acc(y_fieldmap[:, 4], GTMap)


        print('finished')
        f = open('pred_map.txt', 'a+')
        for i in range(len(pred_map)):
            for j in range(len(pred_map[i])):
                f.write(str(pred_map[i][j]) + ',')
            f.write('\n')
        f.close()
        f = open('true_map.txt', 'a+')
        for i in range(len(true_map)):
            for j in range(len(true_map[i])):
                f.write(str(true_map[i][j]) + ',')
            f.write('\n')
        f.close()
        yieldmap = list(y_map)
        f = open('yield_map.txt', 'a+')
        for i in range(len(yieldmap)):
            for j in range(len(yieldmap[i])):
                f.write(str(yieldmap[i][j]) + ',')
            f.write('\n')
        f.close()

        y = y_map[:, 4]
        c = 0
        for i in y:
            if i != 0:
                c += 1

        print(c)
        # y_predmap = get_kernelmap(101, 0)
        # t_predmap = get_t_kernelmap(y_predmap)
        # count = show_acc(list(y_map[:, 4]), t_predmap, count)
        print(count)
        print("end")
