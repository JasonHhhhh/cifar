# author: Jason Howe
import os
import tensorflow as tf
from six.moves import xrange
import cifar10_input

data_dir = 'F:/cifar-10-batches-bin'
filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]
for f in filenames:
    if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)

# Create a queue that produces the filenames to read.
filename_queue = tf.train.string_input_producer(filenames)
result_original = cifar10_input.read_cifar10(filename_queue)
print(result_original.uint8image.get_shape)
print(filenames)
# with tf.Session() as sess:
#     print(sess.run(result_original.label[0]))
