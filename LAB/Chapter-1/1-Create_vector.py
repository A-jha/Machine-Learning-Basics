from numpy.linalg import det
from scipy import sparse
import numpy as np

#===================#
#   Create Vector   #
#===================#

# create a vector as a row
vector_row = np.array([1, 2, 3])
print(vector_row)
"""
[1 2 3]
"""


# create a vector as column
vector_col = np.array([[1], [2], [3]])
print(vector_col)
"""
[[1]
 [2]
 [3]]
"""

#===================#
#   Create Matrix   #
#===================#

matrix_obj = np.mat([[1, 2], [1, 2], [1, 2]])
print(matrix_obj)
"""
[[1 2]
 [1 2]
 [1 2]]
"""

#======================#
# Create Sparse Matrix #
#======================#

# to create sparse matrix we have to import scipy library
# to istall scipy 'sudo pip istall scipy'
#from scipy import sparse

matrix = np.array([[0, 0], [0, 1], [3, 0]])

# create compressed sparse row (CSR) marix
matrix_spars = sparse.csr_matrix(matrix)
print(matrix_spars)
"""
(1, 1)--index        1 -- non-zero-at-11
(2, 0)        3
"""

# Create larger matrix
matrix_large = np.array([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
matrix_large_sparsh = sparse.csr_matrix(matrix_large)
print(matrix_large_sparsh)
"""
  (0, 5)        4
  (1, 1)        1
  (2, 0)        3
"""

#======================#
#  Selecting Elements  #
#======================#

# create a row  vector
vector = np.array([1, 2, 3, 4, 5, 6])

# select element at index 1
print(vector[1])
# ---> 2


# select all elements
print(vector[:])
# ---> [1 2 3 4 5 6]


# select upto and including 3rd element
print(vector[:3])
# ---> [1 2 3]


# select everything after the third element
print(vector[3:])
# ---> [4 5 6]


# select last element
print(vector[-1])
# ---> 6


# create a matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# select second row second col
print(matrix[1][1])
# ---> 5

print(matrix[1, 1])
# ---> 5


# select first 2 row and all col of matrix
print(matrix[:2, :])
"""
[[1 2 3]
 [4 5 6]]
"""

# select all row and second column
print(matrix[:, 1:2])
"""
[[2]
 [5]
 [8]]
"""

#========================#
#   Describing a Matrix  #
#=======================#

matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# sahpe of matrix
print(matrix.shape)
# ---> (3, 4)


# views number of element row * col
print(matrix.size)
# ---> 12


# views number of dimentions
print(matrix.ndim)
# ---> 2


#=======================#
# Operations On element #
#=======================#

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


# create a fuction that adds 100 to something
def add_100(i): return i+100


# create vectorized function
vectorized_add_100 = np.vectorize(add_100)

# apply function to all element of the matrix
matrix_100 = vectorized_add_100(matrix)

print(matrix_100)
"""
[[101 102 103]
 [104 105 106]
 [107 108 109]]
"""

# Or simply we can add 100
print(matrix+100)
"""
[[101 102 103]
 [104 105 106]
 [107 108 109]]
"""

#====================#
# Max and Min Values #
#====================#

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# max element in matrix
print(np.max(matrix))
# ---> 9

# min element in matrix
print(np.min(matrix))
# ---> 1


# Find maximum element in each column
print(np.max(matrix, axis=0))
# ---> [7 8 9]


# find min element in each row
print(np.min(matrix, axis=1))
# ---> [1 4 7]


#====================================#
# average varience and Std Deviation #
#====================================#
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Mean of Matrix
print(np.mean(matrix))
# ---> 5

# varience of matrix
print(np.var(matrix))
# ---> 6.666666666666667


# Standard deviation of Matrix
print(np.std(matrix))
# ---> 2.581988897471611


# mean of each col
print(np.mean(matrix, axis=0))
# ---> [4. 5. 6.]


#===================#
#  Reshaping Array  #
#===================#

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# make matrix of 2 x 6
matrix_reshaped = matrix.reshape(2, 6)
print(matrix_reshaped)
"""
[[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]]
"""

# pass -1 in reshape
Row_Matrix = matrix.reshape(1, -1)
print(Row_Matrix)
# ---> [[ 1  2  3  4  5  6  7  8  9 10 11 12]]


#========================#
#  Transposing A Matrix  #
#========================#

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# transpose the matrix
transposed_mat = matrix.T
print(transposed_mat)
"""
[[1 4 7]
 [2 5 8]
 [3 6 9]]
"""

# transpose a row matrix
row_mat = matrix.reshape(1, -1)
print(row_mat)
# ---> [[1 2 3 4 5 6 7 8 9]]
print(row_mat.T)
"""
[[1]
 [2]
 [3]
 [4]
 [5]
 [6]
 [7]
 [8]
 [9]]
"""

#======================#
# Flattening a Matrix  #
#======================#
matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# Flatten the Matrix
flat_mat = matrix.flatten()
print(flat_mat)
# ---> [1 2 3 4 5 6 7 8]


#=================#
# Rank Of Matrix  #
#=================#
"""
The rank of a matrix is the dimensions of the vector space spanned by its
columns or rows.
"""
matrix = np.array([[1, 1, 1], [1, 1, 10], [1, 1, 15]])

# rank of matrix
rank = np.linalg.matrix_rank(matrix)
print(rank)
# ---> 2


#==========================#
#  Determinant of a Matrix #
#==========================#

matrix = np.array([[1, 2, 1], [1, 2, 1], [101, 111, 15]])

# return determinant of matrix
det_mat = np.linalg.det(matrix)
print(det_mat)
# ---> 0


#========================#
#  Diagonal Of A matrix  #
#========================#

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# return diagonal of matrix
print(matrix.diagonal())
# ---> [1, 5, 9]

# diagonal one above the main diagonal
print(matrix.diagonal(offset=1))
# ---> [2, 6]

# diagonal one below the main diaginal
print(matrix.diagonal(offset=-1))
# ---> [4, 8]


#==================#
# Trace of Matrix  #
#==================#
"""
The trace of a matrix is the sum of the diagonal elements and is often used under
the hood in machine learning methods.
"""
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(matrix.trace())
# ---> 15

#==============================#
# Eigenvalues and Eigenvectors #
#==============================#


matrix = np.array([[1, -1, 3], [1, 1, 6], [3, 8, 9]])

# calculate eigenvalue and eigenvector
eigenvalue, eigenvector = np.linalg.eig(matrix)

print(eigenvalue)
# ---> [13.55075847  0.74003145 -3.29078992]

print(eigenvector)
"""
[[-0.17622017 -0.96677403 -0.53373322]
 [-0.435951    0.2053623  -0.64324848]
 [-0.88254925  0.15223105  0.54896288]]
"""

#===============#
#  Dot Product  #
#===============#
vector_A = np.array([1, 2, 3])
vector_B = np.array([4, 5, 6])

# dot product of A.B
dot_AB = np.dot(vector_A, vector_B)
print(dot_AB)
# ---> 32


# or we can use A@B for dot product
print(vector_A @ vector_B)
# ---> 32


#===========================#
# Addition and Subtraction  #
#===========================#

a = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
b = np.array([[1, 3, 1], [1, 3, 1], [1, 3, 8]])

# Add A and B Matrix
print(np.add(a, b))  # a+b
"""
[[2 4 2]
 [2 4 2]
 [2 4 9]]
"""

# subtract two matrix
print(np.subtract(a, b))
"""
[[ 0 -2  0]
 [ 0 -2  0]
 [ 0 -2 -7]]
"""

#=====================#
# Multiplying Matrix  #
#=====================#
matrix_a = np.array([[1, 1], [1, 2]])

matrix_b = np.array([[1, 3], [1, 2]])

# Multiply two matrices
print(np.dot(matrix_a, matrix_b))
"""
[[2 5]
 [3 7]]
"""

# for element wise multiplication
print(matrix_a*matrix_b)
"""
[[1 3]
 [1 4]]
"""

#=====================#
# Inverting A matrix  #
#=====================#

matrix = np.array([[1, 4], [2, 5]])

print(np.linalg.inv(matrix))
