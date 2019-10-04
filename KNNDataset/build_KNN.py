import csv
import sys

truth_input = sys.argv[1]
pred_input = sys.argv[2]
neighbor_input = sys.argv[3]

truth = []
pred = []
neighbor = []
output = []

with open(truth_input, 'r') as data:
    reader = csv.reader(data)
    for row in reader:
        truth.append(row)

with open(pred_input, 'r') as data:
    reader = csv.reader(data)
    for row in reader:
        pred.append(row)

with open(neighbor_input, 'r') as data:
    reader = csv.reader(data)
    for row in reader:
        neighbor.append(row)

for i in range(len(truth)):
    print([len(truth[i]),len(pred[i])])

#print(truth)
#print(pred)
#print(neighbor)
