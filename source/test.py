import tensorflow as tf
import time
import csv

import vgg19
from model import ASCNet
from config import config
from utils import write_logs, img_read, img_write

height = config.IMG.height
width = config.IMG.width
gamma = config.TRAIN.gamma

def test_datapath():
	list_fr1 = []
	list_fr2 = []
	with open('..\\data\\test\\test.csv') as video:
		reader = csv.DictReader(video)
		s = 1
		for row in reader:
			scene_begin = int(row['scene_begin'])
			scene_end = int(row['scene_end'])
			for fr in range(scene_begin, scene_end-1, 2):
				fr1_path = "..\\data\\test\\{}\\{:05d}.png".format(s, fr)
				fr2_path = "..\\data\\test\\{}\\{:05d}.png".format(s, fr+2)
				list_fr1.append(fr1_path)
				list_fr2.append(fr2_path)
			s = s + 1
		video.close()
	return list_fr1, list_fr2

def test_parse(fr1_dir, fr2_dir):
	fr1_str = tf.read_file(fr1_dir)
	fr2_str = tf.read_file(fr2_dir)

	fr1, h, w = tf.py_func(img_read, [fr1_str], [tf.uint8, tf.int32, tf.int32])
	fr2, _, _ = tf.py_func(img_read, [fr2_str], [tf.uint8, tf.int32, tf.int32])

	fr1 = tf.image.convert_image_dtype(fr1, tf.float32)
	fr2 = tf.image.convert_image_dtype(fr2, tf.float32)

	fr1 = tf.reshape(fr1, [h, w, 3])
	fr2 = tf.reshape(fr2, [h, w, 3])

	return fr1, fr2

def testing(intp_dir, logs_dir, model_dir):
	I1 = tf.placeholder(tf.float32, [1, height, width, 3])
	I2 = tf.placeholder(tf.float32, [1, height, width, 3])

	# Prediction
	with tf.name_scope('ASCNet'):
		I = ASCNet(I1, I2, reuse=False)

	I_ = tf.clip_by_value(I, 0.0, 1.0)
	I_ = tf.multiply(I_, 255.0)
	I_ = tf.cast(I_, tf.uint8)

	test_fr1, test_fr2 = test_datapath()
	num_test = len(test_fr1)
	test_data = tf.data.Dataset.from_tensor_slices((test_fr1, test_fr2))
	test_data = test_data.map(test_parse, num_parallel_calls=4)
	test_data = test_data.batch(1)
	test_iter = test_data.make_one_shot_iterator()
	fr1, fr2 = test_iter.get_next()

	saver = tf.train.Saver(tf.global_variables())

	with tf.Session() as sess:
		# Initialize
		sess.run(tf.global_variables_initializer())

		# Restore weights of model
		saver.restore(sess, model_dir)

		# testation
		log = "\n========== Test Begin ==========\n"
		write_logs(logs_dir, log, True)
		test_start = time.time()
		for path in test_fr1:
			test_fr_start = time.time()
			F1, F2 = sess.run([fr1, fr2])
			_, intp_fr = sess.run([I, I_], feed_dict={I1:F1, I2:F2})

			_, _, _, scene, frame = path.split("\\")
			fr_id, _ = frame.split(".")
			fr_id = int(fr_id) + 1
			frame = "{:05d}".format(fr_id) + ".png"
			fr_dir = intp_dir + "\\" + scene + "\\" + frame
			img_write(fr_dir, intp_fr, 'PNG-FI')

			test_fr = scene + '/' + frame
			log = "Image {}, Time {:2.5f}, Shape = {}"\
						.format(test_fr, time.time()-test_fr_start, intp_fr.shape)
			write_logs(logs_dir, log, False)

		log = "\nTesting Time: {:2.5f}".format(time.time()-test_start)
		write_logs(logs_dir, log, False)
		log = "\n========== Test End ==========\n"
		write_logs(logs_dir, log, False)
		sess.close()



