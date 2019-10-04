import sys
import csv

truth_file = sys.argv[1]
pred_file = sys.argv[2]
neigh_file = sys.argv[3]

truth = []
pred = []
neigh = []
output = []

with open(truth_file) as data:
    reader = csv.reader(data, delimiter=' ')
    for row in reader:
        row = [int(float(i)) for i in row]
        truth.append(row)

with open(pred_file) as data:
    reader = csv.reader(data, delimiter=' ')
    for row in reader:
        row = [float(i) for i in row]
        pred.append(row)

with open(neigh_file) as data:
    reader = csv.reader(data, delimiter=' ')
    for row in reader:
        row = [int(float(i)) for i in row]
        neigh.append(row)


for i in range(len(truth)):
    line = truth[i]
    for j in pred[i]:
        line.append(j)
    line.append(neigh[i][2])
    line.append(neigh[i][4])
    line.append(neigh[i][6])
    line.append(neigh[i][8])
    output.append(line)

outfile = sys.argv[4]
with open(outfile, "w") as of:
    wr = csv.writer(of, quoting=csv.QUOTE_MINIMAL)
    for data in output:
        wr.writerow(data)
