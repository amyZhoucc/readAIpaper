import pickle
file = open("out.pkl","rb")
data = pickle.load(file)
print(data)