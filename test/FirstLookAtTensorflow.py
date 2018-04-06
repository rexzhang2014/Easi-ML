
# coding: utf-8

# Getting Started with Tensorflow 
# 
# Notes: 
# 1. constant: 可以代入公式的常数
# 2. placeholder: 运行时需要传入的数值，用于传入训练样本(X, Y)
# 3. variable: 公式中的变量，用于需要训练来确认的值如W
# 
# For More Reference : https://tensorflow.google.cn/get_started/get_started

# In[2]:


import tensorflow as tf
import pandas as pd
import numpy as np


# <!--img src="http://www.forkosh.com/mathtex.cgi? \Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}"-->
# 1. DEFINE constant
# 2. Run session

# In[8]:


#Create constants
node1 = tf.constant(3.0, dtype=tf.float32) 
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2) # Print out Tensor at this moment

# Create 'add' computation
from __future__ import print_function
node3 = tf.add(node1, node2) #
print("node3:", node3)

# Create Session
sess = tf.Session() 

# Run Session
result = sess.run([node1,node2])
print(type(result), result)
#print(sess.run([node1, node2])) # Print the run result

print("sess.run(node3):", sess.run(node3))


# 1. Define placeholder : to create the computation graph but provide value at session run-time
# 2. Run session

# In[18]:


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

#Tensor could be a number
print(sess.run(adder_node, {a: 3, b: 4.5}))

# A vector
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# A matrix
print(sess.run(adder_node, {a: [[1, 3], [1, 1]], b: [[2, 4], [5, -1]]}))


# Define Variables: Variables 指的是可以通过模型进行训练的那些参数。前面的constant是预先定好的，placeholder是在执行时要传入的。
# 

# In[25]:


# 指定线性模型的参数为W和b，而x为自变量，函数形式为Wx + b

# 假设我们已知一个线性模型的参数值，可令W和b为定值， 再将x作为placeholder在run时传入， 则可以得到模型的输出值
W = tf.Variable([.3], dtype=tf.float64) # 如果使用tf.float32可以观察到0.3000001的数值
b = tf.Variable([-.3], dtype=tf.float64)
x = tf.placeholder(tf.float64)
linear_model = W*x + b 

# 初始化：
# 对constant，创建时进行初始化，所以可以直接run
# 对variable，创建时不初始化，因此在run之前要先初始化
init = tf.global_variables_initializer() # 注意这里用创建Variable时定义的初始值进行初始化
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))


# 1. 定义constant，placeholder及variable
# 2. 定义Loss/Cost/Objective function：通常我们只知道函数的结构（表达式），但并不能确定哪个参数最好，则需要通过样本来拟合最合适的参数值； 目标函数是解决机器学习问题的关键，它刻画的是模型分布和数据分布之间的差异，当这个差异最小时，说明我们的模型分布和数据分布最为接近； 目标函数直接决定了如何训练，若目标函数为凸函数，则极值唯一存在，若目标函数可微，则可以使用梯度算法。常用的目标函数是最小二乘和交叉熵。
# 3. 进行训练： 训练算法是当今机器学习研究的热点，由于不同目标函数的性质，只能使用在这种目标下适用的算法。当模型结构十分复杂时，需要考虑两点，一是是否有足够的数据量来调整大量的参数；二是算法的性能必须使其能在可接受的时间范围内完成，性能有两种场景需要考虑，一是实时应用必须有非常快的计算，二是在开发调整模型的时候，需要算法足够快以实现快速迭代的开发模式。

# In[28]:


# 定义目标函数： 平方和误差，这个误差函数的最小值是0
y = tf.placeholder(tf.float64)
squared_deltas = tf.square(linear_model - y)  #前面定义了linear_model的表达式
loss = tf.reduce_sum(squared_deltas)  #求平方和
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})) # 观察loss的值，发现它是一个较大的值

# 现在改变W和b的值，再次观察loss值， 注意此时不能改变(x,y)，否则验证失去意义
fixW = tf.assign(W, [-1.]) # assign修改变量的值， 返回的是一个handle，可以传给session在run时去执行
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


# 1. 创建训练算法对象： 机器学习中最最常见和简单的训练算法是梯度下降。注意梯度下降要求目标函数可导。若目标函数为凸函数，一定可以收敛到极值点，否则不一定收敛，也不一定收敛到全局最优。
# 2. 执行训练

# In[29]:



optimizer = tf.train.GradientDescentOptimizer(0.01) #创建梯度下降对象，Learning rate=0.01
train = optimizer.minimize(loss) # 创建优化器对象， 目标函数为前面定义的loss对象

sess.run(init) # 用前面的定义的初始化对象进行初始化

# 执行1000次迭代进行训练， 每次迭代更新一次参数
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}) 

print(sess.run([W, b]))

