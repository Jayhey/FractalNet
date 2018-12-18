import numpy as np
import tensorflow as tf
import sys


def recursive(num, _list):
	_list.append(num)
	if num % 2 == 1:
		return recursive(num // 2, _list)
	else:
		return _list


def at_least_one_true(count):
	arr = [1] + [0 for _ in range(count - 1)]
	np.random.shuffle(arr)
	return np.array(arr)


def whether_drop(b, c, is_global, local_rate, deep=None):
	if deep is True:
		return np.ones((b, c, 2 ** (c - 1))) == 1
	else:
		row_num = 2 ** (c - 1)
		rate_holder = np.zeros(shape=[c, row_num])

		def block_drop():
			for row in range(row_num):
				row_list = recursive(row, [])
				if len(row_list) > 1:
					join_rate = np.random.binomial(1, 1 - local_rate, len(row_list))
					# 모두 0일 때는 적어도 하나는 살려라
					if np.any(join_rate) is False:
						join_rate = at_least_one_true(len(row_list))
					for col in range(len(row_list)):
						rate_holder[col][row] = join_rate[col]
				else:
					rate_holder[0][row] = 1
			return rate_holder

		if is_global is True:
			global_path = np.where(at_least_one_true(c) == 1)[0]
			rate_holder[global_path] = 1
			return np.array([rate_holder for _ in range(b)]) == 1
		else:
			return np.array([block_drop() for _ in range(b)]) == 1


def set_input(img, label):
	img /= 255
	img_tensor = tf.constant(img, dtype=tf.float32)
	label_tensor = tf.constant(label, dtype=tf.float32)
	# img_tensor = tf.image.convert_image_dtype(img, dtype=tf.float32)
    # label_tensor = tf.constant(label)
	return img_tensor, label_tensor


def set_drop(b, c, imgs, batch_size, test=False):
	mini_batch = len(imgs) // batch_size + 1
	drop = np.empty((0, b, c, 2 ** (c - 1)), dtype=bool)
	# is_training = "test" if test else "train"

	for i in range(mini_batch):
		global_bool = (i % 2 == 0)
		tmp_whether = whether_drop(b, c, is_global=global_bool, local_rate=0.15, deep=test)
		drop = np.append(drop, [tmp_whether], axis=0)
	#     sys.stdout.write('\rPreprocess %s drop path : %.2f%%' % (is_training, i*100/mini_batch))
	#     sys.stdout.flush()
	# print('\rPreprocess %s drop path complete' % (is_training))
	return drop


class FractalBlock:
	def __init__(self, x, c, block_num, filters, drop_holder, drop_rate):
		with tf.name_scope('block_{}'.format(block_num)):
			self.x = x
			self.block_num = block_num
			self.filters = filters
			self.drop_holder = drop_holder
			self.drop_rate = drop_rate
			self.tensor_col = {_: [self.x] for _ in range(c)}
			row_num = 2 ** (c - 1)
			for row in range(row_num):
				row_list = recursive(row, [])
				if len(row_list) > 1:
					join_conv = self._join_operation(self.tensor_col, row_list)
					for i in range(len(row_list)):
						self.tensor_col[i].append(join_conv)

				elif len(row_list) == 1:
					tmp_conv = self._fractal_conv(self.tensor_col[0][row_list[0]], filters, 0, row)
					self.tensor_col[0].append(tmp_conv)

	def _fractal_conv(self, input_, filters, col, row):
		with tf.name_scope('column_{}_{}'.format(col, row)):
			conv = tf.layers.Conv2D(filters, kernel_size=3, padding='same')(input_)
			conv = tf.cond(self.drop_holder[self.block_num][col][(2**col)*(row+1)-1],
			               lambda: tf.layers.dropout(conv, rate=self.drop_rate),
			               lambda: tf.layers.dropout(conv, rate=1.))
			conv = tf.layers.BatchNormalization()(conv)
			conv = tf.nn.relu(conv)
		return conv

	def _join_operation(self, tensor_col, row_list):
		join = []
		for col, r in enumerate(row_list):
			join.append(self._fractal_conv(tensor_col[col][r], self.filters, col, r))
		join = tf.reduce_mean(join, 0)
		return join

# def set_input(img, label, b, c, test=False):
#     np.random.seed(1234)
#     idx = np.random.permutation(len(img))
#     drop = np.empty((0, b, c, 2**(c-1)), dtype=bool)
#     is_training = "test" if test else "train"

#     for i in range(len(idx)):
#         global_bool = (i%2==0)
#         tmp_whether = whether_drop(b, c, is_global=global_bool, local_rate=0.15, deep=test)
#         drop = np.append(drop, [tmp_whether], axis=0)
#         sys.stdout.write('\rPreprocess %s tensor : %.2f%%' % (is_training, i*100/len(idx)) )
#         sys.stdout.flush()

#     img = img[idx]/255
#     label = label[idx]

#     img_tensor = tf.constant(img)
#     label_tensor = tf.constant(label)
#     drop_tensor = tf.constant(drop, dtype=tf.bool)
#     sys.stdout.write("\rPreprocess %s tensor complete. " % (is_training))
#     return img_tensor, label_tensor, drop_tensor
