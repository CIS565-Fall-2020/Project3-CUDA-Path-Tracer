import csv
import matplotlib.pyplot as plt

data = []
with open('path-termination.csv', 'r') as csvfile:
  spamreader = csv.reader(csvfile, delimiter=',')
  for row in spamreader:
    drow = []
    for data_str in row:
      drow.append(int(data_str))
    data.append(drow)

fig, ax = plt.subplots()

for i in range(0, len(data)):
  row = data[i]
  x = [j for j in range(0, len(row))]
  plt.plot(x, row)

plt.title("Change of paths in first 10 iterations")
plt.xlabel("Depth")
plt.ylabel("Path count")
plt.show()