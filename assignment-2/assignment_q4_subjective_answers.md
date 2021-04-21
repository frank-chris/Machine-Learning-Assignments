# ES654-2020-21 Assignment 2

*Chris Francis* - *18110041*

------
The function I implemented works on 1-D numpy arrays as indicated in the README and also additionally handles 2-D numpy arrays as shown in the example below:

Examples showing the working of PolynomialFeatures.transform() 

(Output of q4_pf_test.py)
```
Input 1-D array:
[1 2]

Output 1-D array(degree = 2):
[1. 1. 2. 1. 4.]

Input 2-D array:
[[0 1 2]
 [3 4 5]
 [6 7 8]]

Output 2-D array(degree = 2):
[[ 1.  0.  1.  2.  0.  1.  4.]
 [ 1.  3.  4.  5.  9. 16. 25.]
 [ 1.  6.  7.  8. 36. 49. 64.]]
```

