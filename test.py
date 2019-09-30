import numpy as np


def get_kernelmap(edge_len, zone_idx, height, width):
    kernel_map = []
    if edge_len % 2 != 1:
        print('Length of the kernel size must be odd.')
        exit()
    w = zone_idx % width
    h = zone_idx // width
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
        ep_h = height - 1
    print(w, h, sp_w, sp_h, ep_w, ep_h)
    while sp_h <= ep_h:
        temp = np.arange(sp_h * width + sp_w, sp_h * width + ep_w)
        kernel_map.append(list(temp))
        sp_h += 1
    kernel_map = np.array(kernel_map)
    return kernel_map


def avail_nextstep(zone_idx, kernelmap, width):
    neighbors = []
    neighbors.append(zone_idx - 1)
    neighbors.append(zone_idx + 1)
    neighbors.append(zone_idx - width)
    neighbors.append(zone_idx + width)
    for i in neighbors:
        if i in kernelmap:
            pass
        else:
            neighbors.remove(i)
    return neighbors


point = get_kernelmap(15, 1342, 32, 42)
# print(np.where(point == 1343))
print(point)
nei = avail_nextstep(1342, point, 42)
print(nei)
# np.savetxt('point', point)