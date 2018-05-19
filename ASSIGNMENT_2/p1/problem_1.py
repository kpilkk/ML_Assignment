
# coding: utf-8

# # <center> Problem_1 </center>

# **Dataset Description**: Wisconsin Diagnostic Breast Cancer(WDBC) dataset from the UCI repository.<br>
# * Each row in the dataset represents a sample of biopsied tissue. The tissue for each sample is imaged and 10 characteristics of the nuclei of cells present in each image are characterized. <br>
# * These characteristics are: *Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Number of concave portions of contour, Symmetry, Fractal dimension.*<br> 
# * Each sample used in the dataset is a feature vector of length 30. 
#         * The first 10 entries in this feature vector are the mean of the characteristics listed above for each image.  
#         * The second 10 are the standard deviation and 
#         * last 10 are the largest value of each of these characteristics present in each image.

# **Panda** has been imported for reading `csv` file and it reads data in Dataframe object.<br>
# **numpy** has been imported for linear algebra calculations.

# In[1]:


import pandas as pd
import numpy as np


# Here **check_output** has been used to know about the files present in the `P1_data`.

# In[2]:


from subprocess import check_output
print(check_output(['ls','/home/kk/Desktop/Assignment_2/P1_data/']).decode('utf8'))


# **read_csv** is used to read `csv` file and **.head()** is used to display first 5 rows of the Dataframe.<br>
# As we don't have column names, that's why `header=None` is used, otherwise first row will be interpreted as column names.

# In[3]:


train = pd.read_csv('P1_data/trainX.csv',header = None)
train_label = pd.read_csv('P1_data/trainY.csv',header=None)
train.head()


# **DecisionTreeClassifier** has been imported from **sklearn.tree** to learn *Decision Tree*.<br>
# **accuracy_score** has been imported from **sklearn.metrics** to know *accuracy* after predicting on test data.

# In[4]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
classifier = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=50)


# First decision tree is learned using **.fit** function of **DecisionTreeClassifier** <br>
# Here **decision_tree** stores the learned Decision Tree and using **export_graphviz** it's been exported to `dot` file.

# In[5]:


from sklearn.tree import export_graphviz
decision_tree = classifier.fit(train,train_label)
export_graphviz(decision_tree,out_file='tree.dot') 
#pred_train = classifier.predict(train)


# If getting error in next code install dependency by running following code in terminal `conda install -c conda-forge libiconv`

# Here, `dot -Tpng tree.dot -o decision_tree.png` si used to convert **dot** file to `png` image format.<br>
# **check_output** has been imported from subprocess which executes terminal command.<br>
# After executing this command **decision_tree.png** image is created which is our learned decision tree.

# In[6]:


print(check_output(['dot','-Tpng','tree.dot', '-o', 'decision_tree.png']).decode('utf8'))


# ### (a) Plot of decision tree model

# **%matplotlib inline** is used to plot the tree on the notebook itself.<br>
# `pyplot` from `matplotlib` has been imported to plot graph.<br>
# **mpimg** has been imported to read image.<br>
# `plt.rcParams` is used to set figure size.

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.rcParams["figure.figsize"] = (15,15)
img=mpimg.imread('decision_tree.png')
plt.imshow(img)


# ### (b) Total number of Nodes in the tree
# So, the **total number of nodes** we have is **33**

# **node_count** is used to count the total number of nodes in the tree

# In[8]:


decision_tree.tree_.node_count


# ### (c) Total number of leaf nodes in the tree
# As we can see **total number of leaf node** is **17**.

# In[9]:


total_nodes=decision_tree.tree_.node_count
left_children = decision_tree.tree_.children_left
right_children = decision_tree.tree_.children_right

features_as_leaves = np.zeros(shape=total_nodes, dtype=bool)

stack = [(0, -1)]  # seed is the root node id and its parent depth
no_of_leaves=0
while len(stack) > 0:
    node_id, parent_depth = stack.pop()

    # If we have a test node
    if (left_children[node_id] != right_children[node_id]):
        stack.append((left_children[node_id], parent_depth + 1))
        stack.append((right_children[node_id], parent_depth + 1))
    else:
        features_as_leaves[node_id] = True
        no_of_leaves = no_of_leaves + 1
print("Total no of leaf nodes are : {}".format(no_of_leaves))


# test data and its label are read in `test` and `test_label`.

# In[10]:


test = pd.read_csv('P1_data/testX.csv',header = None)
test_label = pd.read_csv('P1_data/testY.csv',header=None)
test.head()


# **acuracy_test** is a function to determine the accuracy taking n as the input which is the number of training examples corresponding to 10-100%.<br>
# In `pandas` **iloc** or **loc** is used to access rows and columns.

# In[11]:


def accuracy_test(n):
    classifier.fit(train.iloc[0:n],train_label[0:n])
    test_pred = classifier.predict(test)
    accuracy=accuracy_score(test_label,test_pred)*100
    return accuracy


# **(Q)** (ii) Train your binary decision tree with increasing sizes of training set, say 10%, 20%, ..., 100%.
# and test the trees with the test set. Make a plot to show how training and test accuracies vary with
# number of training samples.

# ### Accuaracy on test_data considering 10-100% of train data.

# In[12]:


for i in range(1,11,1):
    print('Considerin train_set of {}%, Accuracy is {}'.format(i*10,format(accuracy_test(int(len(train)*i/10)))))


# **Accuracy on traing over all training data.**

# In[13]:


pred_label = classifier.predict(test)
accuracy_score(test_label,pred_label)*100


# **test_accuracy** and **percentage_of_train** are lists which stores accuracies corresponding to respective train percentage of train data.

# In[14]:


test_accuracy = []
percentage_of_train = []
for i in range(10,110,10):
    accu = accuracy_test(int(len(train)*i/100))
    test_accuracy.append(accu)
    percentage_of_train.append(i)


# In[15]:


test_accuracy


# **Plot to show how training and test accuracies vary with number of training samples.** 

# In[16]:


plt.rcParams["figure.figsize"] = (5,5)
plt.plot(percentage_of_train,test_accuracy)


# **Confusion Matrix**

# In[17]:


from sklearn.metrics import confusion_matrix
confusion_matrix(test_label,pred_label)


# **Misclassification rate of 1st class.**

# In[18]:


4/(28+4)


# **Misclassification rate of 2nd class.**

# In[19]:


4/(21+4)

