import pandas as pd
import numpy as np
# import seaborn as sn
import matplotlib as plt
import PIL
from PIL import Image
from matplotlib import image
import os
from matplotlib import image
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pandas as pd


datadir = 'archive/train/'

metadataset = pd.read_csv("archive/train_annotations.csv")

# load image as pixel array
data = image.imread('archive/train/a44afa4fa73d/403614c1f263/6489245ff976.jpg')
# summarize shape of the pixel array
print(data.dtype)
print(data.shape)
# display the array of pixels as an image
pyplot.imshow(data)
pyplot.show()

loaded_images = np.array([])


for filename in metadataset['rel_image_path']:
    img_data = image.imread(datadir + filename)
	img_data = Image.fromarray(img_data).resize((25,25))
	if loaded_images.size == 0:
		loaded_images = np.array([list(img_data.getdata())])
	else:
		loaded_images = np.vstack([loaded_images,list(img_data.getdata())])
	print('> loaded %s %s' % (filename, img_data.size))


# load image as pixel array
data = image.imread('archive/train/a44afa4fa73d/403614c1f263/6489245ff976.jpg')
data = Image.fromarray(data).resize((25,25))
# summarize shape of the pixel array
print(data.size)
# display the array of pixels as an image
pyplot.imshow(data)
pyplot.show()

metadataset['Negative for Pneumonia'].to_csv("pixel-data-y.csv",index=True )
np.savetxt("pixel-data-x.csv",loaded_images,delimiter=',')

# Importing the dataset
X = pd.read_csv('pixel-data-x.csv',header=None)
y = pd.read_csv('pixel-data-y.csv')
# X = dataset.iloc[:, :].values
y = y.iloc[:, -1].values

# visualising the trees
from sklearn.tree import DecisionTreeClassifier,plot_tree
tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf.fit(X, y)
plot_tree(tree_clf);

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Measure Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# 10- fold Cross validation by moving 30% into test test.

# Splitting the dataset into the Training set and Test set for 30 %
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# Splitting the dataset into the Training set and Test set for 60 %
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.60, random_state = 0)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# Extractivng TP, FP, FN, FP from the CM
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]

# Measure Accuracy
acc = (TP+TN)/(TP+FP+FN+TN)
print("Accuracy: "+str(acc))

#  TP Rate
tpr = (TP)/(TP+FN)
print("True Positive Rate : "+str(tpr))

#  FP Rate
fpr = (FP)/(TN+FP)
print("False Positive Rate : "+str(fpr))

# Precesion, Recall and f1 score.
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# Precession
pre= (TP)/(TP+FP)
print("Precision :"+ str(pre))

# Recall
pre= (TP)/(TP+FP)
print("Precision :"+ str(pre))

# f1 score
f1 = (2*(TP))/((2*TP)+FN+FP)
print("F1 Score :"+ str(f1))

# ROC Curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, y_pred)







# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()