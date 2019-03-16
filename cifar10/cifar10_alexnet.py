from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cifar10_input
import cifar10
import numpy as np
import time
import math
import tensorflow as tf


max_steps =10

batch_size = 100

data_dir = 'F:/cifar/'



images_train,labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)  #数据增强

images_test,labels_test = cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)

x = tf.placeholder(tf.float32,[batch_size,24,24,3])
y_ = tf.placeholder(tf.int32, [batch_size])

#用于显示网络每一层网络的尺寸#
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def weight_variable(shape):#weight变量，制造一些随机噪声来打破完全对称
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):#bias变量，增加了一些小的正值（0.1），避免死亡节点
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)




parameters = []

# conv1
with tf.name_scope('conv1') as scope:#将scope内生成的Variable自动命名为conv1/xxx，便于区分不同卷积层之间的组件
    kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')#初始化卷积核参数
    conv = tf.nn.conv2d(x, kernel, [1, 2, 2, 1], padding='SAME')#对输入的images完成卷积操作
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')#biases初始化为0
    bias = tf.nn.bias_add(conv, biases)#将参数conv和偏置biases加起来
    conv1 = tf.nn.relu(bias, name=scope)#用激活函数relu对结果进行非线性处理
    print_activations(conv1)#打印该层结构
    parameters += [kernel, biases]#将这一层可训练的参数kernel、biases添加到parameters中


# pool1
lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1') #对前面输出的tensor conv1进行LRN处理 但会降低反馈速度
pool1 = tf.nn.max_pool(lrn1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')
print_activations(pool1)

# conv2
with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')#通道数为上一层卷积核数量(所提取的特征数)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
print_activations(conv2)

# pool2
lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
pool2 = tf.nn.max_pool(conv2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID',name='pool2')
print_activations(pool2)

# conv3
with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv3)

# conv4
with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv4)

# conv5
with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv5)

# pool5
pool5 = tf.nn.max_pool(conv5,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID',name='pool5')
print_activations(pool5)

flattened = tf.reshape(pool5, [batch_size, -1])#对pool5输出的tensor进行变形，将其转化为1D的向量，然后连接全连接层
dim = flattened.get_shape()[1].value
# dense1
with tf.name_scope('dense1') as scope:
    W_dense1 = weight_variable([dim, 4096])#将6x6x256个神经元与4096个神经元(输出)全连接
    b_dense1 = bias_variable([4096])
    dense1 = tf.nn.relu(tf.matmul(flattened, W_dense1) + b_dense1, name=scope)#当然也可以用上述卷积层用的bias_add函数
    parameters += [W_dense1, b_dense1]
    weight_loss = tf.multiply(tf.nn.l2_loss(W_dense1),0.4,name='weight_loss') #正则化处理
    tf.add_to_collection('losses',weight_loss)
    print_activations(dense1)


# drop1
keep_prob = tf.placeholder(tf.float32) 
drop1 = tf.nn.dropout(dense1, keep_prob, name='drop1')
print_activations(drop1)


# dense2
with tf.name_scope('dense2') as scope:
    W_dense2 = weight_variable([4096, 4096])
    b_dense2 = bias_variable([4096])
    dense2 = tf.nn.relu(tf.matmul(drop1, W_dense2) + b_dense2, name=scope)
    parameters += [W_dense2, b_dense2]
    weight_loss = tf.multiply(tf.nn.l2_loss(W_dense2),0.4,name='weight_loss') #正则化处理
    tf.add_to_collection('losses',weight_loss)
    print_activations(dense2)

# drop2
drop2 = tf.nn.dropout(dense2, keep_prob, name='drop2')
print_activations(drop2)


# dense3(softmax)
with tf.name_scope('dense3') as scope:
    W_dense3 = weight_variable([4096, 10])###最后输出为1000类
    b_dense3 = bias_variable([10])
    dense3 = tf.nn.bias_add(tf.matmul(drop2, W_dense3), b_dense3, name=scope)#当然也可以用上述卷积层用的bias_add函数
    parameters += [W_dense3, b_dense3]
    print_activations(dense3)

'''
#第一层 —— 卷积层
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层 —— 卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#第三层 —— 全连接层 1024个隐含节点
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#第四层 —— Dropout层 减轻过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#第五层 —— Softmax层 得到最后的概率输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
'''

'''定义损失函数cross entropy，优化器使用Adam'''
cross_entropy1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dense3,labels=y_))

# tf.add_to_collection('losses',cross_entropy)   #tf.add_to_collection：把变量放入一个集合
# cross_entropy1 = tf.add_n(tf.get_collection('losses'),name='total_loss')  #tf.get_collection：从一个集合中取出全部变量，生成一个列表

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy1)

'''定义评测准确率的操作'''
#correct_prediction = tf.equal(tf.argmax(dense3,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#tf.train.start_queue_runners()
#
#init = tf.global_variables_initializer()
#开始训练#
#sess = tf.InteractiveSession()
#tf.global_variables_initializer().run()
#for i in range(20000):
#  batch = mnist.train.next_batch(50)
#
#  if i%2 == 0:
#    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
#    print("step %d, training accuracy %g"%(i, train_accuracy))
#  train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
#
#print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))



#with tf.Session() as sess:
#    sess.run(init)
##    saver.restore(sess,r'.\MNIST_data\save.ckpt-40')
#    
#    for epoch in range(41):
##        sess.run(tf.assign(lr,0.001*(0.95**epoch)))
#        for batch in range(100):
#            
#            images_train,labels_train = sess.run([images_train,labels_train])
#
#            sess.run(train_step,feed_dict = {x:images_train,y_:labels_train,keep_prob:1.0})
##        if epoch%10==0:
##            saver.save(sess,r'F:\Python\MNIST_data\save.ckpt',global_step=epoch)
#        images_test,labels_test = sess.run([images_test,labels_test])
#    
#        test_acc = sess.run(accuracy,feed_dict = {x:images_test,y_:labels_test,keep_prob:1.0})
##        train_acc = sess.run(accuracy,feed_dict = {x :mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
#        print('Iter',epoch,'Testing Accuracy',test_acc)
        
    
top_k_op = tf.nn.in_top_k(dense3,y_,1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
     
tf.train.start_queue_runners()

true_count = 0

for step in range(max_steps):
    start_time = time.time()
    image_batch,label_batch = sess.run([images_train,labels_train])
    sess.run(train_step,feed_dict={x:image_batch,y_:label_batch,keep_prob:0.5})
    loss1 = sess.run(cross_entropy1,feed_dict={x:image_batch,y_:label_batch,keep_prob:1.0})
    predictions = sess.run(top_k_op,feed_dict={x:image_batch,y_:label_batch,keep_prob:1.0})
    # print(predictions)
    true_count = np.sum(predictions)
    # print(true_count)

    prediction = true_count / batch_size
    
    duration = time.time() - start_time
    
    
    
    
    if step % 1 == 0:
        examples_per_sec = batch_size/duration
        sec_per_batch = float(duration)
#        format_str = ('step ,loss=%.2f (%.1f examples/sec;%.3f sec/batch)')
        print(step,examples_per_sec,sec_per_batch,loss1,prediction)






#import cifar10,cifar10_input
#import tensorflow as tf
#import numpy as np
#import time
#import math
#       
num_examples = 10000
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
true_count1 = 0
total_sample_count = num_iter*batch_size
step = 0
while step < num_iter:
    image_batch,label_batch = sess.run([images_test,labels_test])
    predictions = sess.run([top_k_op],feed_dict={x:image_batch,y_:label_batch,keep_prob:1.0})
    true_count = np.sum(predictions)
    step += 1
    prediction = true_count / batch_size
    true_count1 += true_count
    print('prediction @ 1 = %.3f' % prediction)



    
prediction = true_count1 / total_sample_count
print('prediction @ 1 = %.3f' % prediction)    
        
        
        
