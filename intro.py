
# coding: utf-8

# # Homework 1

# This homework is intended as a brief overview of the machine learning process and the various topics you will learn in this class. We hope that this exercise will allow you to put in context the information you learn with us this semester. Don't worry if you don't understand the techniques here (that's what you'll learn this semester!); we just want to show you how you can use sklearn to do simple machine learning. 

# ## Setup

# First let us import some libraries.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


# For this homework assignment, we will be using the MNIST dataset. The MNIST data is a collection of black and white 28x28 images, each picturing a handwritten digit. These were collected from digits people write at the post office, and now this dataset is a standard benchmark to evaluate models against used in the machine learning community. This may take some time to download. If this errors out, try rerunning it.

# In[2]:


mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64')
y = mnist.target.astype('int64')


# ## Data Exploration

# Let us first explore this data a little bit.

# In[3]:


print(X.shape, y.shape) 


# The X matrix here contains all the digit pictures. The data is (n_samples x n_features), meaning this data contains 70,000 pictures, each with 784 features (the 28x28 image is flattened into a single row). The y vector contains the label for each digit, so we know which digit (or class - class means category) is in each picture.

# Let's try and visualize this data a bit. Change around the index variable to explore more.

# In[10]:


index = 1982 #15000, 28999, 67345
image = X[index].reshape((28, 28))
plt.title('Label is ' + str(y[index]))
plt.imshow(image, cmap='gray')


# Notice that each pixel value ranges from 0-255. When we train our models, a good practice is to *standardize* the data so different features can be compared more equally. Here we will use a simple standardization, squeezing all values into the 0-1 interval range.

# In[5]:


X = X / 255


# When we train our model, we want it to have the lowest error. Error presents itself in 2 ways: bias (how close our model is to the ideal model), and variance (how much our model varies with different datasets). If we train our model on a chunk of data, and then test our model on that same data, we will only witness the first type of error - bias. However, if we test on new, unseen data, that will reflect both bias and variance. This is the reasoning behind cross validation.

# So, we want to have 2 datasets, train and test, each used for the named purpose exclusively.

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# ## Applying Models

# Now we will walk you through applying various models to try and achieve the lowest error rate on this data.

# Each of our labels is a number from 0-9. If we simply did regression on this data, the labels would imply some sort of ordering of the classes (ie the digit 8 is more of the digit 7 than the digit 3 is, etc. We can fix this issue by one-hot encoding our labels. So, instead of each label being a simple digit, each label is a vector of 10 entries. 9 of those entries are zero, and only 1 entry is equal to one, corresponding to the index of the digit. Let's take a look.

# In[12]:


enc = OneHotEncoder(sparse=False)
y_hot = enc.fit_transform(y.reshape(-1, 1))
y_train_hot = enc.transform(y_train.reshape(-1, 1))
y_hot.shape


# Remember how the first sample is the digit zero? Let's now look at the new label at that index.

# In[13]:


y_hot[0]


# ### Linear Regression

# There are 3 steps to build your model: create the model, train the model, then use your model to make predictions). In the sklearn API, this is made very clear. First you instantiate the model (constructor), then you call train it with the `fit` method, then you can make predictions on new data with the `test` method.

# First, let's do a basic linear regression.

# In[14]:


linear = LinearRegression()
linear.fit(X_train, y_train_hot)


# In[15]:


# use trained model to predict both train and test sets
y_train_pred = linear.predict(X_train)
y_test_pred = linear.predict(X_test)

# print accuracies
print('train acc: ', accuracy_score(y_train_pred.argmax(axis=1), y_train))
print('test acc: ', accuracy_score(y_test_pred.argmax(axis=1), y_test))


# Note on interpretability: you can view the weights of your model with `linear.coef_`

# ### Ridge Regression

# Let us try and regularize by adding a penalty term to see if we can get anything better. We can penalize via the L2 norm, aka Ridge Regression.

# In[20]:


ridge = Ridge(alpha=0.3)
ridge.fit(X_train, y_train_hot)
print('train acc: ', accuracy_score(ridge.predict(X_train).argmax(axis=1), y_train))
print('test acc: ', accuracy_score(ridge.predict(X_test).argmax(axis=1), y_test))


# The alpha controls how much to penalize the weights. Play around with it to see if you can improve the test accuracy.

# Now you have seen how to use some basic models to fit and evaluate your data. You will now walk through working with more models. Fill in code where needed.

# ### Logistic Regression

# We will now do logistic regression. From now on, the models will automatically one-hot the labels (so we don't need to worry about it).

# In[21]:


logreg = LogisticRegression(C=0.01, multi_class='multinomial', solver='saga', tol=0.1)
logreg.fit(X_train, y_train)
print('train acc: ', accuracy_score(logreg.predict(X_train), y_train))
print('test acc: ', accuracy_score(logreg.predict(X_test), y_test))


# Our accuracy has jumped ~5%! Why is this? Logistic Regression is a more complex model - instead of computing raw scores as in linear regression, it does one extra step and squashes values between 0 and 1. This means our model now optimizes over *probabilities* instead of raw scores. This makes sense since our vectors are 1-hot encoded.

# The C hyperparameter controls inverse regularization strength (inverse for this model only). Reguralization is important to make sure our model doesn't overfit (perform much better on train data than test data). Play around with the C parameter to try and get better results! You should be able to hit 92%.

# ### Random Forest

# Decision Trees are a completely different type of classifier. They essentially break up the possible space by repeatedly "splitting" on features to keep narrowing down the possibilities. Decision Trees are normally individually very week, so we typically average them together in bunches called Random Forest.

# Now you have seen many examples for how to construct, fit, and evaluate a model. Now do the same for Random Forest using the [documentation here](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). You should be able to create one easily without needing to specify any constructor parameters.

# In[44]:


## YOUR CODE HERE - call the constructor
randomf = RandomForestClassifier(n_estimators = 100)

## YOUR CODE HERE - fit the rf model (just like logistic regression)
randomf.fit(X_train, y_train)

## YOUR CODE HERE - print training accuracy
print('train acc: ', accuracy_score(randomf.predict(X_train), y_train))

## YOUR CODE HERE - print test accuracy
print('test acc: ', accuracy_score(randomf.predict(X_test), y_test))


# WOWZA! That train accuracy is amazing, let's see if we can boost up the test accuracy a bit (since that's what really counts). Try and play around with the hyperparameters to see if you can edge out more accuracy (look at the documentation for parameters in the constructor). Focus on `n_estimators`, `min_samples_split`, `max_features`. You should be able to hit ~97%.

# ### SVC

# A support vector classifier is another completely different type of classifier. It tries to find the best separating hyperplane through your data.

# The SVC will toast our laptops unless we reduce the data dimensionality. Let's keep 80% of the variation, and get rid of the rest. (This will cause a slight drop in performance, but not by much).

# In[29]:


pca = PCA(n_components=0.8, whiten=True)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# Great! Now let's take a look at what that actually did.

# In[30]:


X_train_pca.shape


# Remember, before we had 784 (28x28) features! However, PCA found just 43 basis features that explain 80% of the data. So, we went to just 5% of the original input space, but we still retained 80% of the information! Nice.

# This [blog post](http://colah.github.io/posts/2014-10-Visualizing-MNIST/) explains dimensionality reduction with MNIST far better than I can. It's a short read (<10 mins), and it contains some pretty cool visualizations. Read it and jot down things you learned from the post or further questions.

# - "If we think of it this way, a natural question occurs. What does the cube look like if we look at a particular two-dimensional face? Like staring into a snow-globe, we see the data points projected into two dimensions, with one dimension corresponding to the intensity of a particular pixel, and the other corresponding to the intensity of a second pixel. Examining this allows us to explore MNIST in a very raw way." Oh my god this explanation helped clear things out so much. It is very hard to change the conception of dimansionality so forcing myself to interpret them as cubes really helps.
# 
# - I didn't quite understand the angles of looking at data for a 728 dimensional cube and how they are represented with different colors :(
# 
# 
# - Also the text mentions "One really nice property would be if the distances between points in our visualization were the same as the distances between points in the original space. If that was true, we’d be capturing the global geometry of the data." but I don't understand how just saving the values of each dimention between 0-1 can be represented by distance, what does the distance between points in the original space refer to in this case?

# Now let's train our first SVC. The LinearSVC can only find a linear decision boundary (the hyperplane).

# In[31]:


lsvc = LinearSVC(dual=False, tol=0.01)
lsvc.fit(X_train_pca, y_train)
print('train acc: ', accuracy_score(lsvc.predict(X_train_pca), y_train))
print('test acc: ', accuracy_score(lsvc.predict(X_test_pca), y_test))


# SVMs are really interesting because they have something called the *dual formulation*, in which the computation is expressed as training point inner products. This means that data can be lifted into higher dimensions easily with this "kernel trick". Data that is not linearly separable in a lower dimension can be linearly separable in a higher dimension - which is why we conduct the transform. Let us experiment.

# A transformation that lifts the data into a higher-dimensional space is called a kernel. A polynomial kernel expands the feature space by computing all the polynomial cross terms to a specific degree.

# In[34]:


psvc = SVC(kernel='poly', degree=3, tol=0.01, cache_size=4000)
## YOUR CODE HERE
psvc.fit(X_train_pca,y_train)
## YOUR CODE HERE
print("train acc:", accuracy_score(psvc.predict(X_train_pca), y_train))
## YOUR CODE HERE - print test accuracy
print("test acc:", accuracy_score(psvc.predict(X_test_pca), y_test))


# Play around with the degree of the polynomial kernel to see what accuracy you can get.

# The RBF kernel uses the gaussian function to create an infinite dimensional space - a gaussian peak at each datapoint. Now fiddle with the `C` and `gamma` parameters of the gaussian kernel below to see what you can get. [Here's documentation](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

# In[35]:


rsvc = SVC(kernel='rbf', tol=0.01, cache_size=4000)
## YOUR CODE HERE - fit the rsvc model
rsvc.fit(X_train_pca, y_train) 
## YOUR CODE HERE - print training accuracy
print("train acc:", accuracy_score(rsvc.predict(X_train_pca), y_train))
## YOUR CODE HERE - print test accuracy
print("test acc:", accuracy_score(rsvc.predict(X_test_pca), y_test))


# Isn't that just amazing accuracy?

# ## Basic Neural Network

# You should never do neural networks in sklearn. Use Keras (which we will teach you later in this class), Tensorflow, PyTorch, etc. However, in an effort to keep this homework somewhat cohesive, let us proceed.

# Basic neural networks proceed in layers. Each layer has a certain number of nodes, representing how expressive that layer can be. Below is a sample network, with an input layer, one hidden (middle) layer of 50 neurons, and finally the output layer.

# In[36]:


## with layer size of 50

nn = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', verbose=1)
## YOUR CODE HERE - fit the nn
nn.fit(X_train_pca, y_train)
## YOUR CODE HERE - print training accuracy
print("train acc:", accuracy_score(nn.predict(X_train_pca), y_train))
## YOUR CODE HERE - print test accuracy
print("test acc:", accuracy_score(nn.predict(X_test_pca), y_test))


# In[39]:


# layer size 100

nn = MLPClassifier(hidden_layer_sizes=(1000,), solver='adam', verbose=2)
## YOUR CODE HERE - fit the nn
nn.fit(X_train_pca, y_train)
## YOUR CODE HERE - print training accuracy
print("train acc:", accuracy_score(nn.predict(X_train_pca), y_train))
## YOUR CODE HERE - print test accuracy
print("test acc:", accuracy_score(nn.predict(X_test_pca), y_test))


# Fiddle around with the hiddle layers. Change the number of neurons, add more layers, experiment. You should be able to hit 98% accuracy.

# Neural networks are optimized with a technique called gradient descent (a neural net is just one big function - so we can take the gradient with respect to all its parameters, then just go opposite the gradient to try and find the minimum). This is why it requires many iterations to converge.

# ## Turning In

# Convert this notebook to a PDF (file -> download as -> pdf via latex) and submit to Gradescope.
