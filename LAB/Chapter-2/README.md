# Chapter-2 : Laoding Data

### Sklearn

scikit-learn comes with a number of popular datasets for you to use.

- Download sklearn

```bash
pip install -U scikit-learn
```

```Python
from sklearn import datasets
```

### loading dataset

scikit-learn comes with some common datasets
we can quickly load. These datasets are often called “toy” datasets because they are far smaller and cleaner than a dataset we would see in the real world.

- load_boston
  - Contains 503 observations on Boston housing prices. It is a good dataset for exploring regression algorithms.
- load_iris
  - Contains 150 observations on the measurements of Iris flowers. It is a good dataset for exploring classification algorithms.
- load_digits
  - Contains 1,797 observations from images of handwritten digits. It is a good dataset for teaching image classification.

### Loading a CSV File
