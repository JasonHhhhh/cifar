# author: Jason Howe
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

import cifar10_input

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=100,
                    help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default='F:/cifar/',
                    help='Path to the CIFAR-10 data directory.')

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')

FLAGS = parser.parse_args()

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

num_classes = cifar10_input.NUM_CLASSES

def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels



def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

#正则化处理+变量作用域的创建:
#给出变量形状, 名字， 正态分布的标准差， 衰减因子 返回一个创建好的变量（作用域方法）
def variable_with_weight_decay(name, shape, stddev, wd):
    '''cnn权重变量'''
    var = tf.get_variable(
        name = name,
        shape = shape,
        initializer=tf.truncated_normal_initializer(stddev=stddev,
                                                    dtype=tf.float32),
        dtype=tf.float32,)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def show_feature_map(layer, layer_name, num_or_size_splits, axis, max_outputs):
    split = tf.split(layer, num_or_size_splits=num_or_size_splits, axis=axis)
    for i in range(num_or_size_splits):
        tf.summary.image(layer_name + "-" + str(i), split[i], max_outputs)


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

'''网络搭建'''
def inference(images, training):
    '''
    Built your own CNN
    param
    images:
    [batch_size, height, width, depth(num_channels)]
    training:
    boolean True :  we will drop for training (prob = 0.5)
    False: moves after train: loss &  test & prediction...
    return:
    logits: [batch_size, num_classes]
    即：未经过归一化的“概率” 网络里可以不包含softmax操作，在计算loss是包含即可
    换言之：softmax is for training better
    '''

    ## 设置keep_prob for drop
    if training:
        keep_prob = 0.5
    else: #非训练调用所有节点都保留
        keep_prob = 1

    #注意：下文中pool1 2 conv1 2 等等都是以batchsize为单位计算的所以第一维度为batchsize
    #这一点请与 kernel size 区分（kernel即权重，在一个batch中都相同，所以。。。别搞混就好）

    #创建ALEXNET网络:

    '''cnn'''
    #conv1
    #首先定义卷积核规模（4D）， 通过之来定义这一层神经网络的运算
    # tf.summary.image('monitor_raw', tf.strided_slice(images,[0],[1]), max_outputs=1)
    with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE) as scope:
        kernel = variable_with_weight_decay(
            name = 'weights',
            shape = [5,5,3,64], # 前两位为卷积核size（感受野）， 后两位为进出通道数
            stddev = 0.05,
            wd = 0.0)
        #进行卷积操作： 参数依次： 上一层的输出， 定义好的卷积核变量， stride， padding方法
        conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding = 'SAME')
        #定义偏置变量： 规模为输出通道数 1D
        #why?: 偏置至于输出的通道数有关，因为：
        # 输出的每个通道的map都是经过相加得到的，为每一个map规定偏置没有意义
        biases = _variable_on_cpu(
            name = 'biases',
            shape = [64],
            initializer=tf.constant_initializer(0.0))
        # 激活函数的输入
        pre_activation = tf.nn.bias_add(conv, biases)
        # 激活为下一层的feature maps
        #使用了变量作用域 每次将scope.name赋值给本层的计算结果(激活以后的)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        tf.summary.histogram('conv1',conv1)
        # print('it should be : [batch,x,x,64]', conv1)
        # '''把所有通道都加起来 做一个可视化'''
        # kernel_add_channel1 = tf.constant(1, shape = [1, 1, 64, 1], dtype = tf.float32)
        # conv1_add_channel = tf.nn.conv2d(conv1, kernel_add_channel1, [1, 1, 1, 1], padding = 'SAME')
        # tf.summary.image('monitor_conv1_add_channel', tf.strided_slice(conv1_add_channel,[0],[1]), max_outputs=1)
        # print('it should be : [batch,x,x,1]', conv1_add_channel)
    # 池化1 直接呼叫scope下的conv1 名字不一样，肯定可以call
    # （实际上不同scope中的其他参数在此设计下尽管名字相同，但也是不同节点也不会报错，
    # 但最好不要调直接调用）
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
    # '''把所有通道都加起来 做一个可视化'''
    # print('it should be : [batch,x,x,64]', pool1)
    # kernel_add_channel2 = tf.constant(1,  shape = [1, 1, 64, 1], dtype = tf.float32)
    # pool1_add_channel = tf.nn.conv2d(pool1, kernel_add_channel2, [1, 1, 1, 1], padding = 'SAME')
    # tf.summary.image('monitor_pool1_add_channel', tf.strided_slice(pool1_add_channel,[0],[1]), max_outputs=1)
    # print('it should be : [batch,x,x,1]', pool1_add_channel)
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    # tf.summary.image('monitor_pool1', pool1, max_outputs=3)
    # tf.summary.histogram('pool1', pool1)

    #conv2
    with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE) as scope:
        kernel = variable_with_weight_decay(
            name = 'weights',
            shape = [5,5,64,64],
            stddev = 0.05,
            wd = 0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1,1,1,1], padding = 'SAME')
        biases = _variable_on_cpu(
            name = 'biases',
            shape = [64],
            initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        # '''把所有通道都加起来 做一个可视化 进去batch的第一个图片'''
        # kernel_add_channel3 = tf.constant(1,  shape = [1, 1, 64, 1], dtype = tf.float32)
        # conv2_add_channel = tf.nn.conv2d(conv2, kernel_add_channel3, [1, 1, 1, 1], padding = 'SAME')
        # tf.summary.image('monitor_conv2_add_channel',
        #                  tf.strided_slice(conv2_add_channel,[0],[1]),
        #                                   max_outputs=1)
        # print('it should be : [batch,x,x,1]',conv2_add_channel)
        # print(tf.strided_slice(conv2_add_channel,[0],[1]))
        # tf.summary.image('monitor_conv2', conv2, max_outputs=3)
    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')

    # 池化2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')
    tf.summary.histogram('pool2', pool2)
    # '''把所有通道都加起来 做一个可视化'''
    # print('it should be : [batch,x,x,64]', pool2)
    # kernel_add_channel4 = tf.constant(1,  shape = [1, 1, 64, 1], dtype = tf.float32)
    # pool2_add_channel = tf.nn.conv2d(pool2, kernel_add_channel4, [1, 1, 1, 1], padding = 'SAME')
    # tf.summary.image('monitor_pool2_add_channel', tf.strided_slice(pool2_add_channel,[0],[1]), max_outputs=1)

    # tf.summary.histogram('pool2', pool2)
    # tf.summary.image('monitor_pool2', pool2, max_outputs=3)

# #conv3
# with tf.variable_scope('conv3') as scope:
#     kernel = variable_with_weight_decay(
#         name='weights',
#         shape=[3, 3, 192, 384],
#         stddev=0.1,
#         wd=0.0)
#     conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
#     biases = tf.get_variable(
#         name='biases',
#         shape=[384],
#         initializer=tf.constant_initializer(0.0),
#         dtype=tf.float32)
#     pre_activation = tf.nn.bias_add(conv, biases)
#     conv3 = tf.nn.relu(pre_activation, name=scope.name)
#     tf.summary.histogram('conv3', conv3)

    flattened = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = flattened.get_shape()[1].value

    '''全连接(参数量较大)'''

    #dense1
    # 全连接的定义很简单：矩阵乘法加偏置，然后激活函数为输出
    #偏置size同理
    #权重矩阵：列数为前一层神经元的数量 行数为这一层的数量
    with tf.variable_scope('dense1', reuse=tf.AUTO_REUSE) as scope:
        weights = variable_with_weight_decay(
            name='weights',
            shape=[dim, 384],
            stddev=0.04,
            wd=0.004)
        biases = _variable_on_cpu(
            name='biases',
            initializer= tf.constant_initializer(0.1),
            shape = [384])
        dense1 = tf.nn.relu(tf.matmul(flattened,weights) + biases,
                            name = scope.name)
        tf.summary.histogram('dense1', dense1)

    # drop1
    #即随机置零 每一次训练的模型都是简化过的防止过拟合
    #网络变了，bp会被重写
    # drop1 = tf.nn.dropout(dense1, keep_prob, name='drop1')

    #dense2
    with tf.variable_scope('dense2', reuse=tf.AUTO_REUSE) as scope:
        weights = variable_with_weight_decay(
            name='weights',
            shape=[384, 192],
            stddev=0.04,
            wd=0.004)
        biases = _variable_on_cpu(
            name='biases',
            initializer= tf.constant_initializer(0.1),
            shape = [192])
        dense2 = tf.nn.relu(tf.matmul(dense1,weights) + biases,
                            name = scope.name)
        tf.summary.histogram('dense2', dense2)

    # #drop2
    # drop2 = tf.nn.dropout(dense2, keep_prob, name='drop2')

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear', reuse=tf.AUTO_REUSE) as scope:
        weights = variable_with_weight_decay(
            name='weights',
            shape=[192, num_classes],
            stddev=1/192.0,
            wd=0.0)
        biases = _variable_on_cpu(
            name='biases',
            initializer=tf.constant_initializer(0.0),
            shape=[num_classes])
        softmax_linear = tf.nn.relu(tf.matmul(dense2, weights) + biases,
                                    name=scope.name)
        tf.summary.histogram('softmax_linear', softmax_linear)
    # 实际上返回了还未softmax的输出层（输出规模：类别数）
    return softmax_linear

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]): # 在执行完成loss_averages_op] 后才进行下边的事情
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

def loss(logits, labels):
    '''
    return: (Loss + l2norm_items)
    :param logits: output of inference net without softmax
    :param labels: Labels from distorted_inputs or inputs()
    '''

    # 计算这个batch的平均交叉熵
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)


    return tf.add_n(tf.get_collection('losses'))


















