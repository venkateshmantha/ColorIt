import time
from os.path import isdir, join, basename

import tensorflow as tf
import tensorflow.contrib.slim as slim

from dataset.embedding import prepare_image_for_inception, \
    maybe_download_inception, inception_resnet_v2, inception_resnet_v2_arg_scope
from dataset.tfrecords.images.lab_image_record import LabImageRecordWriter
from dataset.tfrecords.images_queue import queue_single_images_from_folder
from dataset.shared import maybe_create_folder, dir_resized
from dataset.shared import progressive_filename_generator
from dataset.tfrecords import batch_operations


class LabImagenetBatcher:
    def __init__(self, inputs_dir: str, records_dir: str,
                 checkpoint_source: str):
        if not isdir(inputs_dir):
            raise Exception('Input folder does not exists: {}'
                            .format(inputs_dir))
        self.inputs_dir = inputs_dir

        # Destination folder
        maybe_create_folder(records_dir)
        self.records_dir = records_dir

        # Inception checkpoint
        self.checkpoint_file = maybe_download_inception(checkpoint_source)

        # Utils
        self._examples_count = 0
        self.records_names_gen = progressive_filename_generator(
            join(records_dir, 'lab_images_{}.tfrecord'))

    def _initialize_session(self, sess):

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, self.checkpoint_file)

    def batch_all(self, examples_per_record):
        operations = self._create_operations(examples_per_record)

        with tf.Session() as sess:
            self._initialize_session(sess)
            self._run_session(sess, operations, examples_per_record)

    def _create_operations(self, examples_per_record):

        image_key, image_tensor, _ = \
            queue_single_images_from_folder(self.inputs_dir)

        # Build Inception Resnet v2 operations using the image as input
    
        img_for_inception = tf.image.rgb_to_grayscale(image_tensor)
        img_for_inception = tf.image.grayscale_to_rgb(img_for_inception)
        img_for_inception = prepare_image_for_inception(img_for_inception)
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            input_embedding, _ = inception_resnet_v2(img_for_inception,
                                                     is_training=False)

        operations = image_key, image_tensor, input_embedding

        return batch_operations(operations, examples_per_record)

    def _run_session(self, sess, operations, examples_per_record):
    
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        self._examples_count = 0

        try:
            while not coord.should_stop():
                self._write_record(examples_per_record, operations, sess)
        except tf.errors.OutOfRangeError:
            pass
        finally:
            # Ask the threads (filename queue) to stop.
            coord.request_stop()
            print('Finished writing {} images in {:.2f}s'
                  .format(self._examples_count, time.time() - start_time))

        # Wait for threads to finish.
        coord.join(threads)

    def _write_record(self, examples_per_record, operations, sess):
        
        results = sess.run(operations)

        # Create a writer to write the images
        with LabImageRecordWriter(next(self.records_names_gen)) as writer:

            for one_res in zip(*results):
                writer.write_image(*one_res)
                if __debug__:
                    print('Written', basename(one_res[0]))

            self._examples_count += len(results[0])
            print('Record ready:', writer.path)


# Run as python3 -m dataset.lab_batch <args>
if __name__ == '__main__':
    import argparse
    from dataset.shared import dir_tfrecord
    from dataset.embedding.inception_utils import checkpoint_url

    default_batch_size = 500

    parser = argparse.ArgumentParser(
        description='Takes the folder containing 299x299 images, extracts '
                    'the inception resnet v2 features from the image, '
                    'serializes the image in Lab space and the embedding and '
                    'writes everything as tfrecords files '
                    'batches on N images')

    parser.add_argument('-i', '--inputs-folder',
                        default=dir_resized,
                        type=str,
                        metavar='FOLDER',
                        dest='inputs')
    parser.add_argument('-o', '--output-folder',
                        default=dir_tfrecord,
                        type=str,
                        metavar='FOLDER',
                        dest='records')
    parser.add_argument('-c', '--checkpoint',
                        default=checkpoint_url,
                        type=str,
                        dest='checkpoint')
    parser.add_argument('-b', '--batch-size',
                        default=default_batch_size,
                        type=int,
                        metavar='N',
                        dest='batch_size')

    args = parser.parse_args()
    LabImagenetBatcher(args.inputs, args.records, args.checkpoint) \
        .batch_all(args.batch_size)
