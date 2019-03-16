# author: Jason Howe
import tensorflow as tf
import my_cifarmodel
import matplotlib.pyplot as plt
import cifar10_input
import numpy as np
import os
import cv2

parser = my_cifarmodel.parser

parser.add_argument('--var_load_dir', type=str, default='F:/cifar/train_summary_mine',
                    help='dir of the training var record which you want to see its performance of the net')

parser.add_argument('--num_pics', type=int, default=3,
                    help='how many pics you want to see the feature maps of them?(less than batch_size)')

parser.add_argument('--test_dict_dir', type=str, default='F:/cifar/cifar-10-python',
                    help='dir of data in py dict')

parser.add_argument('--map_dir', type=str, default='F:/cifar/featured_maps',
                    help='where to save?')

IMAGE_SIZE = 24

def _unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def preprocess_images(array_images):

    '''输入一组图片的数组信息，将其正则化 再以array输出'''
    class results(object):
        pass

    ims_batch_array_norm = np.zeros((FLAGS.num_pics, IMAGE_SIZE, IMAGE_SIZE, 3))
    ims_batch_array_raw = np.zeros((FLAGS.num_pics, IMAGE_SIZE, IMAGE_SIZE, 3))
    for i in range(array_images.shape[0]):
        # print(array_images.shape[0])
        image_tensor = tf.constant(array_images[i])
        # print(array_image.shape)
        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(image_tensor, [3, 32, 32])
        # Convert from [depth, height, width] to [height, width, depth].
        image_tensor0 = tf.transpose(depth_major, [1, 2, 0])
        reshaped_image = tf.cast(image_tensor0, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # Image processing for evaluation.
        # Crop the central [height, width] of the image.
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                               height, width)


        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(resized_image)
        # print('resized_image',resized_image)
        # original_image = tf.transpose(resized_image, [2,0,1])
        # print('original_image',original_image)
        # Set the shapes of tensor.
        float_image.set_shape([height, width, 3])
        # print('float_image',float_image)
        with tf.Session() as sess:
            ims_batch_array_norm[i] = sess.run(float_image)
            # print(ims_batch_array_norm[i].shape)
            ims_batch_array_raw[i] = sess.run(tf.cast(resized_image, tf.uint8))
            # print(ims_batch_array_raw[i].shape)
    ims_batch_array_raw = ims_batch_array_raw.astype(np.uint8)
    # ims_batch_array_raw = cv2.resize(ims_batch_array_raw, (256,256))
    # print(ims_batch_array_raw[1])
    # print(ims_batch_array_raw[1].shape)
    # plt.imshow(a)
    # plt.show()
    # cv2.destroyAllWindows()


    results.norm_image = ims_batch_array_norm
    results.raw_image = ims_batch_array_raw
    return results

def read_from_test_dict_n():
    '''从二进制测试文件随机读n张图片出来'''
    # 用一个对象集成输出
    class result(object):
        pass

    PATH = os.path.join(FLAGS.test_dict_dir, 'test_batch')
    test_dict =_unpickle(PATH)
    pic_codes = np.random.randint(0, test_dict[ b'data'].shape[0], FLAGS.num_pics)
    pic_array = np.array([test_dict[b'data'][i] for i in pic_codes])
    pic_label = np.array([test_dict[b'labels'][i] for i in pic_codes])
    pic_file_name = np.array([test_dict[b'filenames'][i] for i in pic_codes])

    result.pics_num = FLAGS.num_pics
    result.pics_array = pic_array
    result.pics_label = pic_label
    result.pics_filename = pic_file_name

    return result

    # label_bytes = 1  # 2 for CIFAR-100
    # height = 32
    # width = 32
    # depth = 3
    # image_bytes = height * width * depth
    # # Every record consists of a label followed by the image, with a
    # # fixed number of bytes for each.
    # record_bytes = label_bytes + image_bytes
    # reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # reader.read(filename_queue)

def feature_map_npic():

    '''对随机选区test图片提取每层处理之后的特征图'''

    '''正则化作为网络输入（不做任何其他处理，训练时会那样做，评测、测试不会）'''

    images_holder = tf.placeholder(tf.float32, [FLAGS.num_pics, 24, 24, 3])
    # labels_holder = tf.placeholder(tf.int32, [FLAGS.batch_size])

    '''之前训练的模型框架需要搭建好'''
    with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE) as scope:
        kernel = my_cifarmodel.variable_with_weight_decay(
            name = 'weights',
            shape = [5,5,3,64], # 前两位为卷积核size（感受野）， 后两位为进出通道数
            stddev = 0.05,
            wd = 0.0)
        #进行卷积操作： 参数依次： 上一层的输出， 定义好的卷积核变量， stride， padding方法
        conv = tf.nn.conv2d(images_holder, kernel, [1,1,1,1], padding = 'SAME')
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

    with tf.variable_scope('pool1', reuse=tf.AUTO_REUSE) as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name=scope.name)
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

    with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE) as scope:
        kernel = my_cifarmodel.variable_with_weight_decay(
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
        tf.summary.histogram('conv2', conv2)

    with tf.variable_scope('pool2', reuse=tf.AUTO_REUSE) as scope:
        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')

        # 池化2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name=scope.name)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        # 初始化变量 类似激活
        sess.run(tf.global_variables_initializer())

        # 检查是否存在 ckpt 与变量恢复
        if tf.gfile.Exists(FLAGS.var_load_dir):
            ckpt = tf.train.get_checkpoint_state(FLAGS.var_load_dir)
            if ckpt is not None:
                print("init from" + FLAGS.var_load_dir)
                try:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    get_feature_map = True
                except:
                    print('there is a problem on restore ckpt, a new train should start')
                    get_feature_map = False
        else:
            print('no ckpt')
            get_feature_map = False

        # writer = tf.summary.FileWriter(FLAGS.map_dir, sess.graph)

        if get_feature_map:

            #读入一组图片 （随机从测试机中拿出）
            selected_test_image = read_from_test_dict_n()
            print(selected_test_image.pics_array.shape)
            images_array = preprocess_images(selected_test_image.pics_array)
            print(images_array.raw_image.shape)

            '''喂进去的是什么数组！！！！！sess.run后！！！'''

            conv1_Npics = sess.run(conv1,feed_dict= {images_holder:images_array.norm_image})
            conv2_Npics = sess.run(conv2,feed_dict= {images_holder:images_array.norm_image})
            pool1_Npics = sess.run(pool1,feed_dict= {images_holder:images_array.norm_image})
            pool2_Npics = sess.run(pool2,feed_dict= {images_holder:images_array.norm_image})
            # print('3,  size, size, channel_num,',conv1_Npics.shape)


            conv1_Npics_add = np.sum(conv1_Npics, axis=3)
            poo1_Npics_add = np.sum(pool1_Npics, axis=3)
            conv2_Npics_add = np.sum(conv2_Npics, axis=3)
            pool2_Npics_add = np.sum(pool2_Npics, axis=3)

            # print('3, size, size', conv1_Npics_add.shape)

            for i in range(3):

                plt.figure()
                plt.imshow(images_array.raw_image[i])
                figpath = os.path.join(FLAGS.map_dir,'pic%s_raw_image.png' % (i))
                plt.savefig(figpath)

                plt.figure()
                plt.imshow(conv1_Npics_add[i], interpolation = 'gaussian', cmap=plt.cm.jet)
                figpath = os.path.join(FLAGS.map_dir, 'pic%s_conv1_added.png' % (i))
                plt.savefig(figpath)

                plt.figure()
                plt.imshow(poo1_Npics_add[i], interpolation = 'gaussian', cmap=plt.cm.jet)
                figpath = os.path.join(FLAGS.map_dir, 'pic%s_poo1_added.png' % (i))
                plt.savefig(figpath)

                plt.figure()
                plt.imshow(conv2_Npics_add[i], interpolation = 'gaussian', cmap=plt.cm.jet)
                figpath = os.path.join(FLAGS.map_dir, 'pic%s_conv2_added.png' % (i))
                plt.savefig(figpath)

                plt.figure()
                plt.imshow(pool2_Npics_add[i], interpolation = 'gaussian', cmap=plt.cm.jet)
                figpath = os.path.join(FLAGS.map_dir, 'pic%s_pool2_added.png' % (i))
                plt.savefig(figpath)

                # plt.show()

                # coord.request_stop()
                # # And wait for them to actually do it.
                # coord.join()





def main(argv=None):
    if tf.gfile.Exists(FLAGS.map_dir):
        tf.gfile.DeleteRecursively(FLAGS.map_dir)
    tf.gfile.MakeDirs(FLAGS.map_dir)
    feature_map_npic()

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()






