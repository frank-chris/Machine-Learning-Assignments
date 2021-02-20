# Q4

## Learning Time

### Learning Time vs Number of Samples (N)

#### Real Input Real Output

![LTvsN_RIRO](images/LTvsN_RIRO.png)

#### Real Input Discrete Output

![LTvsN_RIDO](images/LTvsN_RIDO.png)

#### Discrete Input Discrete Output

![LTvsN_DIDO](images/LTvsN_DIDO.png)

#### Discrete Input Real Output

![LTvsN_DIRO](images/LTvsN_DIRO.png)

### Learning Time vs Number of Features (M)

#### Real Input Real Output

![LTvsM_RIRO](images/LTvsM_RIRO.png)

#### Real Input Discrete Output

![LTvsM_RIDO](images/LTvsM_RIDO.png)

#### Discrete Input Discrete Output

![LTvsM_DIDO](images/LTvsM_DIDO.png)

#### Discrete Input Real Output

![LTvsM_DIRO](images/LTvsM_DIRO.png)

## Comparison of observed learning time with theoretical learning time

The theoretical time complexity of the learning algorithm of a decision tree is **O(NMD)** where **N** is the number of samples, **M** is the number of features and **D** is the depth of the tree created. **D** is often considered to be **log(N)** on average, leading to an average time complexity of **O(NMlog(N))**. Since I have fixed a max-depth in my experiments, we can consider **D** to be constant, leading to a theoretical time complexity of **O(NM)**. From the **Learning Time vs N plots**, I observed that learning time was linear with respect to **N**, and from the **Learning Time vs M plots**, I observed that learning time was linear with respect to **M**. Thus, my observations agreed with theoretical complexity for learning time(in all 4 cases).

## Prediction Time

### Prediction Time vs Number of Samples (N)

#### Real Input Real Output

![PTvsN_RIRO](images/PTvsN_RIRO.png)

#### Real Input Discrete Output

![PTvsN_RIDO](images/PTvsN_RIDO.png)

#### Discrete Input Discrete Output

![PTvsN_DIDO](images/PTvsN_DIDO.png)

#### Discrete Input Real Output

![PTvsN_DIRO](images/PTvsN_DIRO.png)

### Prediction Time vs Number of Features (M)

#### Real Input Real Output

![PTvsM_RIRO](images/PTvsM_RIRO.png)

#### Real Input Discrete Output

![PTvsM_RIDO](images/PTvsM_RIDO.png)

#### Discrete Input Discrete Output

![PTvsM_DIDO](images/PTvsM_DIDO.png)

#### Discrete Input Real Output

![PTvsM_DIRO](images/PTvsM_DIRO.png)

## Comparison of observed learning time with theoretical learning time

The theoretical time complexity of the prediction algorithm of a decision tree is **O(ND)** where **N** is the number of samples, and **D** is the depth of the tree created. **D** is often considered to be **log(N)** on average, leading to an average time complexity of **O(Nlog(N))**. Since I have fixed a max-depth in my experiments, we can consider **D** to be constant, leading to a theoretical time complexity of **O(N)**. From the **Prediction Time vs N plots**, I observed that prediction time was linear with respect to **N**(for all 4 cases), and from the **Prediction Time vs M plots**, I observed that prediction time was constant with respect to **M** for 2 cases(RIRO, RIDO) and slightly complicated for 2 cases(DIRO, DIDO). It could be due to my depth limit or some other optimisation in the code. Thus, my observations agreed with theoretical complexity for prediction time(in most cases: 6 out of 8 plots).
