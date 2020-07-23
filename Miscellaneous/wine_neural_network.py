
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing


# In[2]:


flags = tf.app.flags
FLAGS = flags.FLAGS

columns = ['Wine'] + ['col' + str(i) for i in range(1,14)]

df_train = pd.read_csv("train_wine.csv", names=columns)
df_test = pd.read_csv("test_wine.csv", names=columns)


# In[3]:


df_train.shape, df_test.shape


# In[4]:


X_train = df_train[df_train.columns[1:14]].values
X_test = df_test[df_test.columns[1:14]].values


# In[5]:


y_train = df_train['Wine'].values-1
y_test = df_test['Wine'].values-1


# In[6]:


sess = tf.InteractiveSession()


# In[7]:


Y_train = tf.one_hot(indices = y_train, depth=3, on_value=1., off_value=0., axis=1 , name = "one_hot_train").eval()
Y_test = tf.one_hot(indices = y_test, depth=3, on_value=1., off_value=0., axis=1 , name = "one_hot_test").eval()


# In[8]:


X_train, Y_train = shuffle (X_train, Y_train)
X_test, Y_test = shuffle (X_test, Y_test)

scaler = preprocessing.StandardScaler()

sc = scaler.fit(X_train)

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


# In[9]:


# Create the model
x = tf.placeholder(tf.float32, [None, 13])
W = tf.Variable(tf.zeros([13, 3]))
b = tf.Variable(tf.zeros([3]))
y = tf.nn.softmax(tf.matmul(x, W) + b)


# In[10]:


# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 3])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)


# In[11]:


# Train
tf.initialize_all_variables().run()

for i in range(100):
    X_train,Y_train =shuffle (X_train, Y_train, random_state=1)
    
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_xs, batch_ys = X_train , Y_train
    
    train_step.run({x: batch_xs, y_: batch_ys})
    
    cost = sess.run (cross_entropy,
                     feed_dict={x: batch_xs, y_: batch_ys})    
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Cost:{0} | Accuracy:{1}".format(cost, accuracy.eval({x: X_test, y_: Y_test})))

