import pickle
from environments.cube3 import Cube3State

state = Cube3State([27, 28, 35, 3, 4, 16, 42, 21, 33, 24, 34, 18, 25, 13, 12, 44, 23, 0, 51, 39, 9, 19, 22, 10, 53, 30, 38, 6, 7, 45, 5, 31, 14, 29, 32, 36, 47, 1, 15, 46, 40, 41, 11, 37, 20, 2, 52, 8, 48, 49, 50, 17, 43, 26])
data = {"states" : [state]}

# with open("data/cube3/test/jetson-test.pkl", "wb") as file:
#     pickle.dump(data, file)

with open("data/cube3/test/data_0.pkl", "rb") as file:
    loaded = pickle.load(file)
    print(loaded["states"][0].colors)
