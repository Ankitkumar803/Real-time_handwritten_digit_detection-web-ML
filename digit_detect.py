
# Scientific and vector computation for python
import numpy as np



all_theta = np.loadtxt('2darray.csv', delimiter=',')
X_test = np.loadtxt('dummy_data.csv',delimiter=',')


def sigmoid(z):
    """
    Compute sigmoid function given the input z.
    
    Parameters
    ----------
    z : array_like
        The input to the sigmoid function. This can be a 1-D vector 
        or a 2-D matrix. 
    
    Returns
    -------
    g : array_like
        The computed sigmoid function. g has the same shape as z, since
        the sigmoid is computed element-wise on z.

    """
    # convert input to a numpy array
    z = np.array(z)
    
    # ====================== YOUR CODE HERE ======================

    g = 1/(1 + np.exp(-z))

    # =============================================================
    return g



def predictOneVsAll(all_theta, X):
    """
    Return a vector of predictions for each example in the matrix X. 
    Note that X contains the examples in rows. all_theta is a matrix where
    the i-th row is a trained logistic regression theta vector for the 
    i-th class. You should set p to a vector of values from 0..K-1 
    (e.g., p = [0, 2, 0, 1] predicts classes 0, 2, 0, 1 for 4 examples) .
    
    Parameters
    ----------
    all_theta : array_like
        The trained parameters for logistic regression for each class.
        This is a matrix of shape (K x n+1) where K is number of classes
        and n is number of features without the bias.
    
    X : array_like
        Data points to predict their labels. This is a matrix of shape 
        (m x n) where m is number of data points to predict, and n is number 
        of features without the bias term. Note we add the bias term for X in 
        this function. 
    
    Returns
    -------
    p : array_like
        The predictions for each data point in X. This is a vector of shape (m, ).
    

    """
    m = X.shape[0];
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(m)

    # Add ones to the X data matrix
    #X = np.concatenate([np.ones((1, 1)), X], axis=1)
    X = np.hstack((1,X))

    # ====================== YOUR CODE HERE ======================

    predict_mat = sigmoid(np.dot(X,all_theta.T))
    p = np.argmax(predict_mat,axis=0)
    
    #proable_pos = np.argmax(p,axis=1)
    
    
    # ============================================================
    return p

# pred = predictOneVsAll(all_theta, X_test)
# print(pred)