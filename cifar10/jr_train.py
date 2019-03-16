# author: Jason Howe
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import my_cifarmodel
import cifar10_input
import os

import time

import numpy as np

import tensorflow as tf

parser = my_cifarmodel.parser

# parser.add_argument('--train_ckpt', type=str,
#                     default='F:/cifar/train.ckpt',
#                     help='Directory where to write event logs and checkpoint.')

parser.add_argument('--train_dir', type=str, default='F:/cifar/train_summary/',
                    help='Directory where to write event logs and checkpoint.')


parser.add_argument('--max_steps', type=int, default=100000,
                    help='Number of batches to run.')

parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

parser.add_argument('--log_frequency', type=int, default=10,
                    help='How often to log results to the console.')




def train():

    '''训练程序'''

    '''1 数据读入（以batch为单位形成batchqueue, 在sess中采用for不断调用，
    但是注意：实际上读取原始数据在另外的线程一直在进行，我们只是从已读好的batch queue中拿出，
    sess下的for循环： 调用sess.run([images_train, labels_train])）'''
    with tf.device('/cpu:0'): #不写这一句不报错 但是读取的数据好像只有一个batch！！！
        images_train, labels_train = my_cifarmodel.distorted_inputs()

    # 贼鸡儿有用！！！！！随着数据流动记录步数:学习率衰减等等。。。
    global_step = tf.Variable(0, trainable=False)
    '''学习步骤：
    网上扒代码，先跑一次，记录数据，
    再自己进行搭建，核对数据看是否正确！！
    否则这错那错会很乱！！！
    （一定要确定你自己的算法是否实现！！！）'''
    # images_train, labels_train = cifar10_input.distorted_inputs(
    #     data_dir=FLAGS.data_dir,batch_size=FLAGS.batch_size)  # 数据增强

    '''2 建立holder（数据入口），后续在会话中传入相应的变量'''
    images_holder = tf.placeholder(tf.float32, [FLAGS.batch_size, 24, 24, 3])
    labels_holder = tf.placeholder(tf.int32, [FLAGS.batch_size])
    # 变量在inference内全部被创建 网络已搭建好
    logits = my_cifarmodel.inference(images_holder,training = True)
    saver = tf.train.Saver()

    loss = my_cifarmodel.loss(logits, labels_holder)
    tf.summary.scalar('loss', loss)

    train_op = my_cifarmodel.train(loss, global_step)

    top_k_op = tf.nn.in_top_k(logits, labels_holder, 1)

    true_rate = tf.reduce_mean(tf.cast(top_k_op, tf.float32))

    tf.summary.scalar('accuracy_rate', true_rate)

    merged = tf.summary.merge_all()

    # 开始训练：
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        # # 协调器
        # coord = tf.train.Coordinator()
        # 启动线程管理器
        print('开始训练')

        tf.train.start_queue_runners()
        # print('列队开启')global_step
        for step in range(FLAGS.max_steps):

            start_time = time.time()

            # if coord.should_stop():
            #     break
            # print('数据读取中。。。')
            images_batch, labels_batch = sess.run([images_train, labels_train])
            # print('数据读取完成，进行训练')
            print('类型', type(images_batch))
            sess.run(train_op,
                     feed_dict={images_holder:images_batch,
                                labels_holder:labels_batch})
            # print('训练完成，计算loss，精度等')
            #计算loss
            loss_value = sess.run(loss,
                                  feed_dict={images_holder: images_batch,
                                             labels_holder: labels_batch})
            #这个batch上成功率为多少（这个值随着迭代而增大）
            true_counter = sess.run(top_k_op,
                                feed_dict={images_holder: images_batch,
                                           labels_holder: labels_batch})
            ##
            accuracy = np.sum(true_counter)/FLAGS.batch_size
            # print('一次训练完成')
            summary = sess.run(merged,
                               feed_dict={images_holder: images_batch,
                                          labels_holder: labels_batch})
            train_writer.add_summary(summary, step) #这一句再写入log

            #持续时间
            duration = time.time() - start_time
            # print('数据记录完成')
            #log
            if step % FLAGS.log_frequency == 0:
                if step % (FLAGS.log_frequency*10) == 0:
                    ckptpath = os.path.join(FLAGS.train_dir,'model.ckpt')
                    saver.save(sess, ckptpath,step)  # 这一句在保存变量
                examples_per_sec = FLAGS.batch_size / duration
                sec_per_batch = float(duration)
                format_str = ('step %d\tloss = %.2f\taccuracy_rate = %.4f\t'
                              'counter=%d\t'
                              '(%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (step, loss_value, accuracy,np.sum(true_counter),
                                    examples_per_sec, sec_per_batch))

        # coord.request_stop()
        # # And wait for them to actually do it.
        # coord.join()
        # # 保存训练结果
        train_writer.close()


def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()

