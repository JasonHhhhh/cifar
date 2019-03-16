# author: Jason Howe
# author: Jason Howe
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import my_cifarmodel
import cifar10_input


import time

import numpy as np

import tensorflow as tf

parser = my_cifarmodel.parser

# parser.add_argument('--train_ckpt', type=str,
#                     default='F:/cifar/train.ckpt',
#                     help='Directory where to write event logs and checkpoint.')

parser.add_argument('--train_dir', type=str, default='F:/cifar/train_summary_mine/',
                    help='Directory where to write event logs and checkpoint.')

parser.add_argument('--test_dir', type=str, default='F:/cifar/train_test_jr/test_log/',
                    help='Directory where to write event logs and checkpoint.')

parser.add_argument('--max_steps', type=int, default=100000,
                    help='Number of batches to run.')

# parser.add_argument('--log_device_placement', type=bool, default=False,
#                     help='Whether to log device placement.')

parser.add_argument('--log_frequency', type=int, default=100,
                    help='How often to log results to the console.')




def train_then_test():

    '''训练程序'''

    '''1 数据读入（以batch为单位形成batchqueue, 在sess中采用for不断调用，
    但是注意：实际上读取原始数据在另外的线程一直在进行，我们只是从已读好的batch queue中拿出，
    sess下的for循环： 调用sess.run([images_train, labels_train])）'''

    # images_train, labels_train = cifar10_input.distorted_inputs(data_dir=FLAGS.data_dir,
    #                                               batch_size=FLAGS.batch_size)
    # 贼鸡儿有用！！！！！随着数据流动记录步数:学习率衰减等等。。。
    '''创建一个变量：初始化操作后为0 每次train_op 之后自动加1 ------> 记录全局步数'''
    global_step = tf.Variable(0, trainable=False)
    '''学习步骤：
    网上扒代码，先跑一次，记录数据，
    再自己进行搭建，核对数据看是否正确！！
    否则这错那错会很乱！！！
    （一定要确定你自己的算法是否实现！！！）'''
    with tf.device('/cpu:0'): #不写这一句不报错 但是读取的数据好像只有一个batch！！！
        #只有在sess后才会读入，没有sess只是一个batchqueue在那里
        '''因为：你建立的都是静止图节点！！！！！！！！！！！！！'''
        images_train, labels_train = my_cifarmodel.distorted_inputs()
        # images_test, labels_test = cifar10_input.inputs(eval_data=True,
        #                                 data_dir=FLAGS.data_dir,
        #                                 batch_size=FLAGS.batch_size)
    # images_train, labels_train = cifar10_input.distorted_inputs(
    #     data_dir=FLAGS.data_dir,batch_size=FLAGS.batch_size)  # 数据增强
    #
    # images_test, labels_test = cifar10_input.inputs(
    #     eval_data=True,data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size)

    '''2 建立holder（数据入口），后续在会话中传入相应的变量'''
    images_holder = tf.placeholder(tf.float32, [FLAGS.batch_size, 24, 24, 3])
    labels_holder = tf.placeholder(tf.int32, [FLAGS.batch_size])

    # images_test_holder = tf.placeholder(tf.float32, [FLAGS.batch_size, 24, 24, 3])
    # labels_test_holder = tf.placeholder(tf.int32, [FLAGS.batch_size])
    # 变量在inference内全部被创建 网络已搭建好
    logits = my_cifarmodel.inference(images_holder,training = True)

    saver_allvar = tf.train.Saver()
    # saver_globstep = tf.train.Saver({'global_step':global_step})

    #训练与测试同步进行 训练后的网络参数可以共享给测试集
    loss = my_cifarmodel.loss(logits, labels_holder)
    # loss_test = my_cifarmodel.loss(logits_test, labels_test_holder)
    tf.summary.scalar('monitor_loss', loss)
    # tf.summary.scalar('loss_test', loss_test)
    top_k_op = tf.nn.in_top_k(logits, labels_holder, 1)
    # top_k_op_test = tf.nn.in_top_k(logits_test, labels_test_holder, 1)
    true_rate = tf.reduce_mean(tf.cast(top_k_op, tf.float32))
    # true_rate_test = tf.reduce_mean(tf.cast(top_k_op_test, tf.float32))

    tf.summary.scalar('monitor_accuracy_rate', true_rate)

    train_op = my_cifarmodel.train(loss, global_step)


    # tf.summary.scalar('accuracy_rate_test', true_rate_test)

    merged = tf.summary.merge_all()
    #数据的每一次流动 我都在记录

    # 开始训练：
    with tf.Session() as sess:
        #变量全部初始化
        sess.run(tf.global_variables_initializer())

        '''此处确定是否有断点文件，若不想延续则需人工删除 可以写一个函数'''
        if tf.gfile.Exists(FLAGS.train_dir):
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt is not None:
                print("init from" + FLAGS.train_dir)
                try:
                    saver_allvar.restore(sess,ckpt.model_checkpoint_path)
                    ## 使summary连续记录 读取全局步数
                    # Assuming model_checkpoint_path looks something like:
                    #  /my-favorite-path/cifar10_train/model.ckpt-0,
                    # extract global_step from it.
                    # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                except:
                    tf.gfile.DeleteRecursively(FLAGS.train_dir)
                    tf.gfile.MakeDirs(FLAGS.train_dir)
                    print('there is a problem on restore ckpt, a new train will start')
        else:
            print('start a new train')
            tf.gfile.MakeDirs(FLAGS.train_dir)

        # if tf.gfile.Exists(FLAGS.test_dir):
        #     ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        #     if ckpt is not None:
        #         print("init from" + FLAGS.train_dir)
        #         try:
        #             saver_allvar.restore(sess,ckpt.model_checkpoint_path)
        #         except:
        #             tf.gfile.DeleteRecursively(FLAGS.train_dir)
        #             tf.gfile.MakeDirs(FLAGS.train_dir)
        #             print('there is a problem on restore ckpt, a new train will start')
        # else:
        #     print('start a new train')
        #     tf.gfile.MakeDirs(FLAGS.train_dir)

        '''写入器定义'''
        writer_train = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        # writer_test = tf.summary.FileWriter(FLAGS.test_dir)
        # global_step = sess.run(global_step)
        '''开始训练'''
        # # 协调器
        # coord = tf.train.Coordinator()
        # 启动线程管理器
        tf.train.start_queue_runners()
        print('start from step:',sess.run(global_step))
        # print('列队开启')global_step
        last_step = sess.run(global_step)  + FLAGS.max_steps
        for step in range(sess.run(global_step), last_step):
            # print('step:', sess.run(global_step))
            # saver_globstep.saver()
            start_time = time.time()

            # if coord.should_stop():
            #     break
            # print('数据读取中。。。')
            images_train_batch, labels_train_batch = sess.run(
                [images_train, labels_train])
            # images_test_batch, labels_test_batch = sess.run(
            #     [images_test, labels_test])

            # images_batch_test, labels_batch_test = sess.run(
            #     [images_test, labels_test])
            # print(images_batch_test,images_batch_train)
            # feed_dict_train = {images_train_holder: images_batch_train,
            # #              labels_train_holder: labels_batch_train}
            # feed_dict_test = {images_holder: images_test_batch,
            #              labels_holder: labels_test_batch}
            feed_dict_train = {images_holder: images_train_batch,
                         labels_holder: labels_train_batch}

            # print('数据读取完成，进行训练')
            sess.run(train_op,feed_dict_train)
            # print('训练完成，计算loss，精度等')
            #计算loss
            loss_value_train = sess.run(loss,feed_dict_train)
            # print(loss_value_train)
            # loss_value_test = sess.run(loss,feed_dict_test)

            #这个batch上成功率为多少（这个值随着迭代而增大）
            true_counter_train = sess.run(top_k_op,feed_dict_train)
            # true_counter_test = sess.run(top_k_op,feed_dict_test)
            #
            accuracy_train = np.sum(true_counter_train) / FLAGS.batch_size
            # accuracy_test = np.sum(true_counter_test) / FLAGS.batch_size
            # print('一次训练完成')
            #通过给网络未如不同的数据，创建不同的写入器，来写入不同的记录
            #
            summary_train = sess.run(merged, feed_dict_train)
            # summary_test = sess.run(merged, feed_dict_test)
            # writer_train.add_summary(summary_train, step) #这一句再写入log
            # writer_test.add_summary(summary_test, step) #这一句再写入log

            # images_test, labels_test = sess.run(
            #     [images_test, labels_test])
            # feed_dict_test = {images_holder: images_test,
            #                   labels_holder: labels_test}
            # loss_value_test = sess.run(loss, feed_dict_test)
            # true_counter_test = sess.run(top_k_op, feed_dict_test)
            # accuracy_test = np.sum(true_counter_test) / FLAGS.batch_size
            # summary_test = sess.run(merged, feed_dict_test)
            # writer_test.add_summary(summary_test, step)
            #持续时间
            duration = time.time() - start_time
            # print('数据记录完成')
            #log
            '''存储全局步数'''
            # saver_allvar.save(sess, FLAGS.train_dir)
            if step % FLAGS.log_frequency == 0:
                # print(sess.run(global_step)-1)
                writer_train.add_summary(summary_train, step)
                print(sess.run(kernel_add_channel3).sum()/64)
                # 这一句在保存变量
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver_allvar.save(sess, checkpoint_path, step)
                '''每n步进行一次测试并且存入指定的目录'''
                expls_per_sec = FLAGS.batch_size / duration
                sec_per_batch = float(duration)
                print('step:', step,
                      'TRAIN:loss:', loss_value_train,
                      'accuracy_rate:', accuracy_train,
                      'counter:', np.sum(true_counter_train),
                      'expls_per_sec', expls_per_sec,
                          'sec_per_batch', sec_per_batch)
                # print('step:', step,
                #       ' TEST:loss:', loss_value_test,
                #       ' accuracy_rate:', accuracy_test,
                #       ' counter:', np.sum(true_counter_test))

                # print(format_str2 % (step, loss_value_test, accuracy_test, np.sum(true_counter_test)))

        # coord.request_stop()
        # # And wait for them to actually do it.
        # coord.join()
        writer_train.close()
        # writer_test.close()


def main(argv=None):
    # if tf.gfile.Exists(FLAGS.test_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.test_dir)
    # tf.gfile.MakeDirs(FLAGS.test_dir)

    train_then_test()

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()

