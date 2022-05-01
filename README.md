This is a Frame
I love Huai Huai

## For Features:

1. Feature Binarization
2. Calculate Discriminate Power of each feature
3. Implement PCA. Observe the relation to Discriminate Power


## For Training and Prediction:
### Regression:
- [ ] SVR   --Cheng
- [ ] Knn   --Cheng
- [ ] 1nn   --Cheng

### Classification:
- [ ] RBF NN    --Huai
- [ ] Random Forest
- [ ] KMeans

## Reference System:
### Regression:
- [ ] Trivial System
    - A system that always outputs the mean output value 𝑦< from the training set.
- [ ] Baseline System
    - 1NN (output value is the same as the nearest training-set data point in feature space)
    - Linear Regression (no regularization)
### Classification:
- [ ] Trivial System
    - A system that randomly outputs class labels with probability based on class priors 
    (priors calculated from the training set). Run the trivial system at least 10 times and take 
    the average of the results (accuracy or macro f1-score) as the 
    final performance of the trivial system.
- [ ] Baseline System
    - Nearest Means