# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""

from datetime import datetime
import time
from tensorflow_examples.cnn import cifar10

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
						   """Directory where to write event logs """
						   """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
							"""Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', True,
							"""Whether to log device placement.""")


def train():
	"""Train CIFAR-10 for a number of steps."""
	with tf.Graph().as_default():
		ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
		global_step_init = -1
		if ckpt and ckpt.model_checkpoint_path:
			# Assuming model_checkpoint_path looks something like:
			#   /my-favorite-path/cifar10_train/model.ckpt-0,
			# extract global_step from it.
			global_step_init = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
			global_step = tf.Variable(global_step_init, name='global_step', dtype=tf.int64, trainable=False)
		else:
			global_step = tf.contrib.framework.get_or_create_global_step()

		# global_step = tf.contrib.framework.get_or_create_global_step()

		# Get images and labels for CIFAR-10.
		images, labels = cifar10.distorted_inputs()

		# Build a Graph that computes the logits predictions from the
		# inference model.
		logits = cifar10.inference(images)

		# Calculate loss.
		loss = cifar10.loss(logits, labels)

		# Build a Graph that trains the model with one batch of examples and
		# updates the model parameters.
		train_op = cifar10.train(loss, global_step)

		class _LoggerHook(tf.train.SessionRunHook):
			"""Logs loss and runtime."""

			def begin(self):
				self._step = global_step_init

			def before_run(self, run_context):
				self._step += 1
				self._start_time = time.time()
				return tf.train.SessionRunArgs(loss)  # Asks for loss value.

			def after_run(self, run_context, run_values):
				duration = time.time() - self._start_time
				loss_value = run_values.results
				if self._step % 10 == 0:
					num_examples_per_step = FLAGS.batch_size
					examples_per_sec = num_examples_per_step / duration
					sec_per_batch = float(duration)

					format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
								  'sec/batch)')
					print(format_str % (datetime.now(), self._step, loss_value,
										examples_per_sec, sec_per_batch))

		saver = tf.train.Saver()
		with tf.train.MonitoredTrainingSession(
				checkpoint_dir=FLAGS.train_dir,
				hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
					   tf.train.NanTensorHook(loss),
					   _LoggerHook()],
				config=tf.ConfigProto(
					log_device_placement=FLAGS.log_device_placement),
				save_checkpoint_secs=120
		) as mon_sess:
			ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(mon_sess, ckpt.model_checkpoint_path)
			while not mon_sess.should_stop():
				mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
	cifar10.maybe_download_and_extract()
	# if tf.gfile.Exists(FLAGS.train_dir):
	#     tf.gfile.DeleteRecursively(FLAGS.train_dir)
	# tf.gfile.MakeDirs(FLAGS.train_dir)
	if not tf.gfile.Exists(FLAGS.train_dir):
		tf.gfile.MakeDirs(FLAGS.train_dir)
	train()


if __name__ == '__main__':
	tf.app.run()
