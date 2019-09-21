
# coding: utf-8

# ## Load MNIST

# In[1]:


import pickle
import gzip

from sklearn import svm
from sklearn.model_selection import GridSearchCV


# In[2]:


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()


# In[3]:


train_feat = training_data[0]
train_tar = training_data[1]
val_feat = validation_data[0]
val_tar = validation_data[1]
test_feat = test_data[0]
test_tar = test_data[1]


# ## SVM Implementation

# In[ ]:


## Finding the best value of 'C' using GridSearchCV. Commented it as it will take a lot of time to run

# # Grid Search to find the best parameters
# # Parameter Grid
# param_grid = {'C': [1, 10, 100]}
 
# # Make grid search classifier
# clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
 
# # Train the classifier
# clf_grid.fit(train_feat, train_tar)
 
# # clf = grid.best_estimator_()
# print("Best Parameters:\n", clf_grid.best_params_)
# print("Best Estimators:\n", clf_grid.best_estimator_)


# In[4]:


C = 2
clf = svm.SVC(kernel='linear', C=C, random_state = 123)
clf.fit(train_feat, train_tar)


# In[5]:


from sklearn.metrics import accuracy_score

# Getting Validation dataset accuracy
val_pred_svm = clf.predict(val_feat)
acc_val_svm = accuracy_score(val_tar, val_pred_svm)

# # Getting Testing dataset accuracy
test_pred_svm = clf.predict(test_feat)
acc_test_svm = accuracy_score(test_tar, test_pred_svm)

print ('---------- Support Vector Machine (SVM) --------------------')
print("Regularization Parameter/Penalty(C) = " + str(C))
print("SVM Validation Accuracy: ", acc_val_svm*100)
print("SVM Test Accuracy: ", acc_test_svm*100)


# ## Lagrange Dual Problem

# $
# {\text{maximize}_\alpha} \sum_i \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j \text{y}_i \text{y}_j \text{x}_i \text{x}_j
# $
# 
# $ \text{subject to:} $
# 
# $
# \sum_i \alpha_i \text{y}_i = 0 \text{ ;}
# $
# 
# $
# 0 \leq \alpha_i \leq \text(C)
# $

# ### The Margin
# $
# \underset{w, b} {\text{argmax}} \Bigg\{ \frac{1}{||w||} \underset{n}{\text{min}} [t_n (w^T \phi(x_n) + b) ] \Bigg\}
# $
# 
# $
# \text{Direct solution to the optimization soultion would be very complex, and so the Margin is converted to Primal optimization form:}
# $
# 
# $
# \underset{w, b} {\text{argmin}} \frac{1}{||w||^2}
# $
# 
# $ \text{subject to:} $
# 
# $
# \{ t_n (w^T \phi(x_n) + b) \} \geq 1
# $
# 
# $ \text {Casting to a unconstrained problem using Lagrange formula and so we wish to Minimize:} $
# 
# $
# L(w,b,a) = \frac{1}{||w||^2} - \sum_{i=1}^n a_n  [t_n (w^T \phi(x_n) + b) - 1];
# \text{where } a = (a_1,..,a_n)^T
# $
# 
# $ \text{Dual Representation is:} $
# 
# $
# L(a) = \sum_{n=1}^n a_n - \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_m a_n \text{t}_m \text{t}_n k(\text{x}_m \text{x}_n)
# $

# ### Benefits of Maximizing the Margin

# In support vector machines the decision boundary is chosen to be the one for which the margin is maximized. The maximum margin solution can be motivated using computational learning theory, also known as statistical learning theory. However, a simple insight into the origins of maximum margin has been given by Tong and Koller (2000) who consider a framework for classification based on a hybrid of generative and discriminative approaches. They first model the distribution over input vectors x for each
# class using a Parzen density estimator with Gaussian kernels having a common parameter σ2. Together with the class priors, this defines an optimal misclassification-rate decision boundary. However, instead of using this optimal boundary, they determine the best hyperplane by minimizing the probability of error relative to the learned density model. In the limit σ2 → 0, the optimal hyperplane is shown to be the one having maximum margin. The intuition behind this result is that as σ2 is reduced, the hyperplane is increasingly dominated by nearby data points relative to more distant ones. In the limit, the hyperplane becomes independent of data points that are not support vectors.
# 
# The marginalization with respect to the prior distribution of the parameters in a Bayesian approach for a simple linearly separable data set leads to a decision boundary that lies in the middle of the region separating the data points. The large margin solution has similar behaviour.

# ### Benefit of Solving the Dual Problem instead of the Primal Problem

# The most significant benefit from solving the dual comes when you are using the "Kernel Trick" to classify data that is not linearly separable in the original feature space.
# 
# The importance of dual formulation is, it lends itself easily to the kernel trick. The optimization problem is almost the same as the original dual formulation, except that you compute the kernel function instead of the Dot-Product. This is not possible in the Primal formulation, where it would be necessary to explicitly compute the mapping for each point(as we can see from Primal optimization equation).
# 
# Also, classifying a new data point becomes much easier. If you solved the primal, you compute
# 
# $f(x) = w^T\phi(x) + w_0$
# 
# for a new data point x, and classify it depending on the sign of the above expression. But if we solve the dual, you would get:
# $f(x) = (\sum_i \alpha_iy_iK(x_i, x)) + w_0$
# 
# So you have Kernel function evaluations instead of explicit mapping computation, and also the fact that most of the \alpha_is would be zero. (They are non-zero only for the "support vectors", which would be few). Bottomline: you classify non-linearly separable data quite easily, with little extra (computational) effort.

# ##### References

# 1. https://www.quora.com/Why-is-solving-in-the-dual-easier-than-solving-in-the-primal-What-advantages-do-we-get-from-solving-in-the-dual
# 
# 2. Pattern Recognition and Machine Learning by Christopher M. Bishop
