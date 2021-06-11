# Chapter 1. Vectors, Matrices, and Arrays

NumPy is the foundation of the Python machine learning stack. NumPy allows
for efficient operations on the data structures often used in machine learning.

### Create a Vector

```Python
vector_row = np.array([1, 2, 3])
print(vector_row)
"""
[1 2 3]
"""
```

### Create A mtrix

```python
matrix_obj = np.mat([[1, 2], [1, 2], [1, 2]])
print(matrix_obj)
"""
[[1 2]
 [1 2]
 [1 2]]
"""
```

## Sparse Matrix

A frequent situation in machine learning is having a huge amount of data;
however, most of the elements in the data are zeros.

imagine amatrix where the columns are every movie on Netflix, the rows are every Netflix user, and the values are how many times a user has watched that particular
movie. This matrix would have tens of thousands of columns and millions of
rows! However, since most users do not watch most movies, the vast majority of
elements would be zero.

Sparse matrices only store nonzero elements and assume all other values will be
zero, leading to significant computational savings.

```Python
print(sparse_mat)
"""
(1, 1)        1
(2, 0)        3
"""
```

The output of sparse matrix shows the index where non zero value is present.

- In above example 1 is present at index 11 and 3 is at index 20

## Eigenvalue and Eigenvector

Eigenvectors are widely used in machine learning libraries. Intuitively, given a
linear transformation represented by a matrix, A, eigenvectors are vectors that,
when that transformation is applied, change only in scale (not direction). More
formally:

<center> Av = λv</center>

where A is a square matrix, λ contains the eigenvalues and v contains the
eigenvectors. In NumPy’s linear algebra toolset, eig lets us calculate the
eigenvalues, and eigenvectors of any square matrix.

## Inverse of a matrix

The inverse of a square matrix, A, is a second matrix A –1.

<center>AA-1 = I</center>
