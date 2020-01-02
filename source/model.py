import tensorflow as tf
import numpy as np

def Create3Conv(x, fs, is_train, name, reuse):
	x = tf.layers.conv2d(x, fs, 3, 1, 'same', activation=tf.nn.relu, trainable=is_train, name='%s/conv1'%name, reuse=reuse)
	x = tf.layers.conv2d(x, fs, 3, 1, 'same', activation=tf.nn.relu, trainable=is_train, name='%s/conv2'%name, reuse=reuse)
	x = tf.layers.conv2d(x, fs, 3, 1, 'same', activation=tf.nn.relu, trainable=is_train, name='%s/conv3'%name, reuse=reuse)
	return x

def UpSampling(x, name):
	_, hx, wx, _ = x.shape
	x = tf.image.resize_bilinear(x, (2*hx, 2*wx), name='%s/upsampling'%name)
	return x

def SkipConnection(x, y):
	bx, hx, wx, cx = x.shape
	by, hy, wy, cy = y.shape
	if hx > hy:
		x = tf.slice(x, [0,0,0,0], [bx, hy, wx, cx])
	elif hx < hy:
		y = tf.slice(y, [0,0,0,0], [by, hx, wy, cy])
	x = x + y
	return x

def ASCNet(I1, I2, is_train=True, reuse=None):
	x = tf.concat([I1, I2], 3)
	# Encoder
	x = Create3Conv(x, 32, is_train, 'encoder32', reuse)
	x = tf.layers.average_pooling2d(x, 2, 2, 'same')
	x64 = Create3Conv(x, 64, is_train, 'encoder64', reuse)
	x = tf.layers.average_pooling2d(x64, 2, 2, 'same')
	x128 = Create3Conv(x, 128, is_train, 'encoder128', reuse)
	x = tf.layers.average_pooling2d(x128, 2, 2, 'same')
	x256 = Create3Conv(x, 256, is_train, 'encoder256', reuse)
	x = tf.layers.average_pooling2d(x256, 2, 2, 'same')
	x512 = Create3Conv(x, 512, is_train, 'encoder512_1', reuse)
	x = tf.layers.average_pooling2d(x512, 2, 2, 'same')
	x = Create3Conv(x, 512, is_train, 'encoder512_2', reuse)

	# Decoder
	x = UpSampling(x, 'decoder512')
	x = SkipConnection(x, x512)
	x = Create3Conv(x, 256, is_train, 'decoder256', reuse)
	x = UpSampling(x, 'decoder256')
	x = SkipConnection(x, x256)
	x = Create3Conv(x, 128, is_train, 'decoder128', reuse)
	x = UpSampling(x, 'decoder128')
	x = SkipConnection(x, x128)
	x = Create3Conv(x, 64, is_train, 'decoder64', reuse)
	x = UpSampling(x, 'decoder64')
	x = SkipConnection(x, x64)

	k1h = Create3Conv(x, 51, is_train, 'k1h', reuse)
	k1h = UpSampling(k1h, 'k1h')

	k1v = Create3Conv(x, 51, is_train, 'k1v', reuse)
	k1v = UpSampling(k1v, 'k1v')

	k2h = Create3Conv(x, 51, is_train, 'k2h', reuse)
	k2h = UpSampling(k2h, 'k2h')

	k2v = Create3Conv(x, 51, is_train, 'k2v', reuse)
	k2v = UpSampling(k2v, 'k2v')

	b, h, w, c = I1.shape
	pv = tf.constant([[0,0], [25,0], [0,0], [0,0]])
	ph = tf.constant([[0,0], [0,0], [25,0], [0,0]])
	F1v = tf.pad(I1, pv, "CONSTANT")
	F2v = tf.pad(I2, pv, "CONSTANT")

	F1v_list = []
	F2v_list = []
	for i in range(51):
		F1v_ = tf.manip.roll(F1v, -i, axis=1)
		F1v_list.append(F1v_)
		F2v_ = tf.manip.roll(F2v, -i, axis=1)
		F2v_list.append(F2v_)

	I1v = tf.stack(F1v_list, axis=3)
	I2v = tf.stack(F2v_list, axis=3)
	I1v = I1v[:,0:h,:,:,:]
	I2v = I2v[:,0:h,:,:,:]

	k1v = tf.expand_dims(k1v, axis=4)
	k1h = tf.expand_dims(k1h, axis=4)
	k2v = tf.expand_dims(k2v, axis=4)
	k2h = tf.expand_dims(k2h, axis=4)

	I1_ = tf.reduce_sum(tf.multiply(k1v, I1v), axis=3)
	I2_ = tf.reduce_sum(tf.multiply(k2v, I2v), axis=3)
	
	F1h = tf.pad(I1_, ph, "CONSTANT")
	F2h = tf.pad(I2_, ph, "CONSTANT")

	F1h_list = []
	F2h_list =[]
	for i in range(51):
		F1h_ = tf.manip.roll(F1h, -i, axis=2)
		F1h_list.append(F1h_)
		F2h_ = tf.manip.roll(F2h, -i, axis=2)
		F2h_list.append(F2h_)
		
	I1h = tf.stack(F1h_list, axis=3)
	I2h = tf.stack(F2h_list, axis=3)
	I1h = I1h[:,:,0:w,:,:]
	I2h = I2h[:,:,0:w,:,:]
	
	I1_ = tf.reduce_sum(tf.multiply(k1h, I1h), axis=3)
	I2_ = tf.reduce_sum(tf.multiply(k2h, I2h), axis=3)
	I = I1_ + I2_
	
	return I






