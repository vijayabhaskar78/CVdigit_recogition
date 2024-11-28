import numpy as np
import cv2

# Loading the digits data
data = cv2.imread('images/digits.png', cv2.IMREAD_COLOR)
if data is None:
    raise FileNotFoundError("The file 'images/digits.png' was not found. Please check the path.")
gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

# Resizing for visualization
resized = cv2.pyrDown(gray)
cv2.imshow("Original Data", resized)
cv2.waitKey(500)
cv2.destroyAllWindows()

# Splitting the image into 5000 sub-images of size 20x20
arr = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
arr = np.array(arr)
print("Resulting Shape", arr.shape)

# Splitting into training and test sets
X_train = arr[:, :70].reshape(-1, 400).astype(np.float32)  # Train: 70 columns
X_test = arr[:, 70:].reshape(-1, 400).astype(np.float32)   # Test: 30 columns
print("Input shapes\n--> Train: {}, Test: {}".format(X_train.shape, X_test.shape))

# Targets for each image
y = np.arange(10)  # 0 to 9
y_train = np.repeat(y, 350)[:, np.newaxis]  # 70% for training
y_test = np.repeat(y, 150)[:, np.newaxis]   # 30% for testing
print("Target shapes\n--> Train: {}, Test: {}".format(y_train.shape, y_test.shape))

# Using K-NN (k-nearest neighbors)
classifier_knn = cv2.ml.KNearest_create()
classifier_knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

# Testing and calculating accuracy
ret, result, neighbours, distance = classifier_knn.findNearest(X_test, k=3)
accuracy = np.mean(result == y_test) * 100
print("Accuracy:", accuracy)
