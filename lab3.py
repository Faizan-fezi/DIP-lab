
# A = np.array([[2, 4, 6], [8, 10, 8], [2, 10, 4]])
# sumOfDiag = np.sum(np.diag(A)) 
# print("Sum of diagonal elements:", sumOfDiag) 

# sumvalue=0
# for i in range(len(A)):
#     sum+=A[i,i]
# print("Sum of diagonals is : ",sum)
# x = float(input('What is the original value? ')) 
# y = x * 10 
# print("Original value:",x)
# print("Value after multiplication by 10:",y)

# import numpy as np

# import numpy as np 
 
# X	= np.array([[1, 2], 
#                 [3, 4]]) 
 
# Y	= np.array([[1, 0], 
#               [0, 1]]) 
 
# Matrix multiplication (standard multiplication) 
# A = np.dot(X, Y) 
# print("Multiplication result: ",A)
 
# Element-wise multiplication 
# B	= X * Y 
 
# # Matrix multiplication with transposed matrix Y 
# C	= np.dot(X, Y.T) 
 
# print("A:") 
# print(A) 
# print("\nB:") 
# print(B) 
# print("\nC:") 
# print(C) 

# import numpy as np 
 
# # Create array similar to (0:2:10) 
# x = np.arange(0, 11, 2).reshape(-1, 1) 
 
# # Perform operations similar to [x x.^2] 
# print("X = \n",x)
# result = np.concatenate((x, x**2), axis=1) 
 
# print(result) 

# import numpy as np 
 
# # Load the matrix from file 
# A = np.loadtxt('matrix.txt') 
 
# # Find determinant of A 
# d = np.linalg.det(A) 
 
# # Find inverse of A 
# Y = np.linalg.inv(A) 
 
# # Multiply A with its inverse 
# result = np.dot(A, Y) 
 
# print("Matrix A:") 
# print(A) 
# print("\nDeterminant of A:", d) 
# print("\nInverse of A:") 
# print(Y) 
# print("\nA multiplied by its inverse:") 
# print(result) 

import numpy as np 
from sklearn import datasets
 
# Load iris dataset 
iris = datasets.load_iris() 
 
# Split the dataset into features (sepal length, sepal width, petal length, petal width) and labels 
features = iris.data
labels = iris.target + 1
 
# Compute mean and standard deviation for each category of iris 
for i in range(1, 4): 
    category_features = features[labels == i]  # Select features for the current category     
    mean_values = np.mean(category_features, axis=0)  # Compute mean along columns     
    std_values = np.std(category_features, axis=0)  # Compute standard deviation along columns 
     
    print(f"\nCategory {i}:")     
    print("Mean values:")     
    print(mean_values)     
    print("Standard deviation:")     
    print(std_values) 

