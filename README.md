
```
conda list --export > requirements.txt
conda create -n <env-name> --file requirements.txt
```


## Getting started with Machine Learning

First, we must clear up one of the biggest misconceptions about machine learning:
- Machine learning is not about algorithms.
- Machine learning is a comprehensive approach to solving problems.

[Data Science Primer ](https://elitedatascience.com/primer)


## Must Read
- [Beginner Mistakes](https://elitedatascience.com/beginner-mistakes)
- [Modern Machine Learning Algorithm: Strengths and Weaknesses](https://elitedatascience.com/machine-learning-algorithms)
- [Fun Machine Learning Projects for Beginners](https://elitedatascience.com/machine-learning-projects-for-beginners)
- [Four-Part Tutorial by Kaggle](https://www.kaggle.com/c/titanic)
- [How to Handle Imbalanced Classes in Machine Learning](https://elitedatascience.com/imbalanced-classes)
- [Datasets for Data Science and Machine Learning](https://elitedatascience.com/datasets)
- [Beginning Kaggle](https://elitedatascience.com/beginner-kaggle)


## Best courses
- [Stanford's Machine Learning Course - Andrew Ng](https://www.youtube.com/watch?v=qeHZOdmJvFU&list=PLZ9qNFMHZ-A4rycgrgOYma6zxF4BZGGPW&index=1)
- [Harvard's Data Science Course](http://cs109.github.io/2015/)


## Libraries to be good at
- Numpy, Pandas, Matplotlib
- [Saeaborn](https://elitedatascience.com/python-seaborn-tutorial)

## Recommended Steps

1. **Exploratory Data Analysis**
	- A quick and dirty grid of histograms.
		- Distributions that are unexpected
		- Potential outliers that don't make sense 
		- Features that should be binary
		- Boundaries that don't make sense
		- Potential measurement errors
	- A bar plot for categorical features.
		- Sparse clases i.e small number of observations
		- Combine / Re-assign sparse classes as part of Feature Scaling
	- Box Plots
		- Median, min, max will help in generalization
	- Study Correlations
		- Positive correlation means that as one feature increases, the other increases.
		- Negative correlation means that as one feature increases, the other decreases. 
		- Correlations near -1 or 1 indicate a strong relationship.
		- Those closer to 0 indicate a weak relationship.
		- 0 indicates no relationship.
	- Check for imbalanced classes
		
2. **Data Cleaning**
	- Remove duplicate and irrelevant observations.
	- Check for mislabeled classes, i.e. separate classes that should really be the same.
	- Remove / Filter outlires, you must have a good reason for removing an outlier tough.
	- Handle Missing Data
		- The best way to handle missing data for categorical features is to simply label them as ’Missing’!
		- For missing numeric data, you should flag the obsearvation with an indicator variable and fill the value as 0. By using this technique of flagging and filling, you are essentially allowing the algorithm to estimate the optimal constant for missingness, instead of just filling it in with the mean.


## Theorotically Learned Algorithms
- Regression
	1. [Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html)
		- lasso
		- Ridge
		- Eastic Net
	2. Decision Trees (Ensembles)
		- [Random Forests(RF)](https://scikit-learn.org/stable/modules/ensemble.html#random-forests) - perform very well out-of-the-box
		- [Gradient Boosted Trees(GBM)](https://scikit-learn.org/stable/modules/ensemble.html#classification) - harder to tune but tend to have higher performance
	3. [Deep Learning (NN)](https://keras.io/)
	4. KNN 
		- memory-intensive, perform poorly for high-dimensional data, and require a meaningful distance function to calculate similarity.
- Classification
	1. [Logistic Regression](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
	2. Decision Trees (Ensembles)
		- [Random Forests(RF)](https://scikit-learn.org/stable/modules/ensemble.html#random-forests) - perform very well out-of-the-box
		- [Gradient Boosted Trees(GBM)](https://scikit-learn.org/stable/modules/ensemble.html#classification) - harder to tune but tend to have higher performance
	3. [Deep Learning (NN)](https://keras.io/)
	4. [SVM](http://scikit-learn.org/stable/modules/svm.html#classification)
	5. [Naiye Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html)
		- It's called "naive" because its core assumption that all input features are independent from one another, rarely holds true in the real world.
- Clustering
	1. [K-Means](http://scikit-learn.org/stable/modules/clustering.html#k-means)
	2. [Affinity Propagation](http://scikit-learn.org/stable/modules/clustering.html#affinity-propagation)
	3. [Hierarchical / Agglomerative](http://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
	4. [DBSCAN](http://scikit-learn.org/stable/modules/clustering.html#dbscan)


TODO:
	imbalanced classes
	Feature Engineering
	Work on some datasets for EDA, data cleaning and Feature Engineering
	Algorithm Selection
	Model Training

	
 
