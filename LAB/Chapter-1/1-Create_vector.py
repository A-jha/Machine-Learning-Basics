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
