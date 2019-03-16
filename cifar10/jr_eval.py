# author: Jason Howe
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import math

import my_cifarmodel
import cifar10_input

parser = my_cifarmodel.parser

parser.add_argument('--eval_dir', type=str,
                    default='F:/cifar/test_log_jr/',
                    help='Directory where to write event logs.')

parser.add_argument('--eval_data', type=str, default='test',
                    help='Either `test` or `train_eval`.')

parser.add_argument('--checkpoint_dir', type=str, default='F:/cifar/train_summary_mine_1/',
                    help='Directory where to read model checkpoints.')

parser.add_argument('--eval_interval_secs', type=int, default=60*5,
                    help='How often to run the eval.')

parser.add_argument('--num_examples', type=int, default=10000,
                    help='Number of examples to run.')

parser.add_argument('--run_once', type=bool, default=False,
                    help='Whether to run eval only once.')



# def eval_once():
#     '''一次测试'''


def eval():
    '''测试主程序'''
    eval_data = FLAGS.eval_data == 'test' # 给出是否适用测试集的布尔变量

    #准备数据，传入数据以batch为单位

    images_test, labels_test = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=FLAGS.data_dir,
                                        batch_size=FLAGS.batch_size)
    # 检测时仍然需要建立holder 节点化仅是数据在流通
    images_holder = tf.placeholder(tf.float32, [FLAGS.batch_size, 24, 24, 3])
    labels_holder = tf.placeholder(tf.int32, [FLAGS.batch_size])
    #网络预测
    logits = my_cifarmodel.inference(images_holder, training = False)
    saver = tf.train.Saver()
    loss = my_cifarmodel.loss(logits, labels_holder)
    tf.summary.scalar('loss', loss)
    #检测效果
    top_k_op = tf.nn.in_top_k(logits, labels_holder, 1)
    true_rate = tf.reduce_mean(tf.cast(top_k_op, tf.float32))
    tf.summary.scalar('true_rate', true_rate)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        test_writer = tf.summary.FileWriter(FLAGS.eval_dir, sess.graph)
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
        total_examples_num = num_iter*FLAGS.batch_size
        true_count = 0
        step = 0
        tf.train.start_queue_runners()
        while step < num_iter:
            #取出一批数据
            images_batch, labels_batch = sess.run([images_test,labels_test])
            #统计预测结果
            # loss_value = sess.run(loss,
            #                       feed_dict={images_holder: images_batch,
            #                          labels_holder: labels_batch})

            true_count0 = np.sum(sess.run(
                top_k_op, feed_dict={images_holder:images_batch,
                                     labels_holder:labels_batch}))
            loss_value, summary = sess.run([loss,merged],
                               feed_dict={images_holder: images_batch,
                                          labels_holder: labels_batch})
            test_writer.add_summary(summary, step)  # 这一句再写入log
            true_rate = true_count0 / FLAGS.batch_size
            step += 1
            true_count += true_count0
            print('step %d true rate: %.3f loss: %.3f' % (step, true_rate, loss_value))
        test_writer.close()
        total_true_rate = true_count / total_examples_num
        print('Total true rate: %.3f' % total_true_rate)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    eval()

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()