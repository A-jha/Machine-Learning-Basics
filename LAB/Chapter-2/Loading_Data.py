# Download sklearn library
import pandas as pd
from sklearn import datasets


#========================================#
# Loading a sample data set from sklearn #
#========================================#
# load scikit learn's datasets
#from sklearn import datasets

# Load Digit datasets
digits = datasets.load_digits()

# create features matrix
features = digits.data

# create target vector
target = digits.target

print(features[0])
"""
[ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
 15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
  0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
  0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]
"""

#===============#
# Laod CSV File #
#===============#
# Pandas library is used to import csv file

# Create Url
url = './MOCK_DATA.csv'
dataframe = pd.read_csv(url)

print(dataframe.head(4))
"""
   Gender  Marks  id
0    True     95   1
1   False     19   2
2    True     84   3
3    True     97   4
"""


#=================#
# Load Excel File #
#=================#
# url = './tt.csv.xlsx'

# dataframe = pd.read_excel(url)

# print(dataframe.head(4))


#================#
# Load JSON File #
#================#
url = './MOCK_DATA.json'

dataframe = pd.read_json(url, orient="columns")

print(dataframe.head(3))
"""
              Card No  Marks  id
0     201475089378062     96   1
1  493673650244060032     19   2
2    5048374579498454     99   3
"""
