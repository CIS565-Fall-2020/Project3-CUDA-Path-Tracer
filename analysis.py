import csv
import matplotlib.pyplot as plt

# data = []
# with open('path-termination.csv', 'r') as csvfile:
#   spamreader = csv.reader(csvfile, delimiter=',')
#   for row in spamreader:
#     drow = []
#     for data_str in row:
#       drow.append(int(data_str))
#     data.append(drow)

# fig, ax = plt.subplots()

# for i in range(0, len(data)):
#   row = data[i]
#   x = [j for j in range(0, len(row))]
#   plt.plot(x, row)

# plt.title("Change of paths in first 10 iterations")
# plt.xlabel("Depth")
# plt.ylabel("Path count")
# plt.show()

# base = [13.9153, 13.4523, 13.2604, 13.0256, 12.8251, 12.6814, 12.584, 12.467, 12.3808,
#         12.3539, 12.3, 12.2559, 12.2064, 12.1793, 12.157, 12.1449, 12.1508, 12.1537, 12.147, 12.152]
# first_cache = [13.994, 13.5278, 12.9571, 12.4803, 12.1568, 11.9892, 11.8137, 11.7125, 11.6294,
#                11.5688, 11.602, 11.5473, 11.5048, 11.4699, 11.4974, 11.476, 11.4706, 11.4416, 11.4178, 11.411]
# ray_term = [43.3666, 39.4445, 37.9706, 37.2061, 36.8238, 36.464, 36.3297, 36.1474, 36.0439,
#             35.9822, 35.937, 35.907, 35.8626, 35.8245, 35.7628, 35.7196, 35.6834, 35.6314, 35.5985, 35.5634]
# mat_sorting = [92.1103, 90.0405, 89.2113, 88.785, 88.5098, 88.3835, 88.2686, 88.1703, 88.1409,
#                88.0665, 88.0275, 88.0738, 88.1269, 88.1731, 88.2425, 88.2707, 88.309, 88.3622, 88.4123, 88.4398]

# x = [i * 50 for i in range(1, 21)]

# fig, ax = plt.subplots()

# plt.plot(x, base, label="base")
# plt.plot(x, first_cache, label="first bounce cache")
# plt.plot(x, ray_term, label="stream compaction")
# plt.plot(x, mat_sorting, label="material sorting")

# plt.title("Average time per iteration for different optimizations")
# plt.xlabel("First # of iterations")
# plt.ylabel("Average time (ms)")
# plt.legend()
# plt.show()

mat_scene_with_sort = [77.1336, 75.3321, 74.7839, 74.586, 74.8918, 75.1525, 74.7427, 74.6345, 74.6205,
                       74.6913, 74.7094, 74.7804, 74.789, 74.7278, 74.6984, 74.7119, 74.7549, 74.8404, 74.8833, 74.86]

mat_scene = [42.1215, 40.9768, 40.5953, 40.4929, 40.7369, 40.6872, 40.7313, 40.7851, 40.8242, 40.67, 40.2865, 39.918, 39.6168, 39.3591, 39.1535, 38.9644, 38.8111, 38.6765, 38.569, 38.458]

x = [i * 50 for i in range(1, 21)]

fig, ax = plt.subplots()

plt.plot(x, mat_scene, label="base")
plt.plot(x, mat_scene_with_sort, label="with material sorting")

plt.title("Average time per iteration for material sorting")
plt.xlabel("First # of iterations")
plt.ylabel("Average time (ms)")
plt.legend()
plt.show()