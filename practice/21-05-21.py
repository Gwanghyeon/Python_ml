from sklearn.preprocessing import Binarizer

x = [[1, -1, 2], [2, 0, 0], [0, 1.1, 1.2]]

binarizer = Binarizer(threshold=1.1)  # return 0 if the value<=threshold
print(binarizer.fit_transform(x))

print(type(binarizer.fit_transform(x)))
