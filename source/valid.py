import tensorflow as tf
import numpy as np
import os
import random
import time
import csv

import vgg19
from model import ASCNet
from config import config
from utils import write_logs, img_read, img_write

height = config.IMG.height
width = config.IMG.width
gamma = config.TRAIN.gamma

def valid_datapath():
	list_fr1 = []
	list_fr2 = []
	list_gt = []
	with open('..\\data\\valid\\valid.csv') as video:
		reader = csv.DictReader(video)
		s = 1
		for row in reader:
			scene_begin = int(row['scene_begin'])
			scene_end = int(row['scene_end'])
			for fr in range(scene_begin, scene_end-1, 2):
				fr1_path = "..\\data\\valid\\{}\\30fps\\val_{:05d}.png".format(s, fr)
				fr2_path = "..\\data\\valid\\{}\\30fps\\val_{:05d}.png".format(s, fr+2)
				gt_path = "..\\data\\valid\\{}\\60fps\\val_{:05d}.png".format(s, fr+1)
				list_fr1.append(fr1_path)
				list_fr2.append(fr2_path)
				list_gt.append(gt_path)
			s = s + 1
		video.close()
	return list_fr1, list_fr2, list_gt

def valid_parse(fr1_dir, fr2_dir, gt_dir):
	fr1_str = tf.read_file(fr1_dir)
	fr2_str = tf.read_file(fr2_dir)
	gt_str = tf.read_file(gt_dir)

	fr1, h, w = tf.py_func(img_read, [fr1_str], [tf.uint8, tf.int32, tf.int32])
	fr2, _, _ = tf.py_func(img_read, [fr2_str], [tf.uint8, tf.int32, tf.int32])
	gt, _, _ = tf.py_func(img_read, [gt_str], [tf.uint8, tf.int32, tf.int32])

	fr1 = tf.image.convert_image_dtype(fr1, tf.float32)
	fr2 = tf.image.convert_image_dtype(fr2, tf.float32)
	gt = tf.image.convert_image_dtype(gt, tf.float32)

	fr1 = tf.reshape(fr1, [h, w, 3])
	fr2 = tf.reshape(fr2, [h, w, 3])
	gt = tf.reshape(gt, [h, w, 3])

	return fr1, fr2, gt

def validate(intp_dir, logs_dir, model_dir):
	I1 = tf.placeholder(tf.float32, [1, height, width, 3])
	I2 = tf.placeholder(tf.float32, [1, height, width, 3])
	Igt = tf.placeholder(tf.float32, [1, height, width, 3])

	# Prediction
	with tf.name_scope('ASCNet'):
		I = ASCNet(I1, I2, reuse=False)

	I_ = tf.clip_by_value(I, 0.0, 1.0)

	valid_fr1, valid_fr2, valid_gt = valid_datapath()
	num_valid = len(valid_fr1)
	valid_data = tf.data.Dataset.from_tensor_slices((valid_fr1, valid_fr2, valid_gt))
	valid_data = valid_data.map(valid_parse, num_parallel_calls=4)
	valid_data = valid_data.batch(1)
	valid_iter = valid_data.make_one_shot_iterator()
	fr1, fr2, gt = valid_iter.get_next()

	with tf.name_scope('L1_Loss'):
		L1 = tf.losses.absolute_difference(Igt, I)

	vgg19_gt = vgg19.Vgg19()
	vgg19_gt.build(Igt)
	vgg19_intp = vgg19.Vgg19()
	vgg19_intp.build(I_)
	with tf.name_scope('Perceptual_Loss'):
		Lf = tf.losses.mean_squared_error(vgg19_gt.conv4_4, vgg19_intp.conv4_4)

	with tf.name_scope('Combined_Loss'):
		Lc = L1 + gamma*Lf
	
	I_ = tf.multiply(I_, 255.0)
	I1_ = tf.multiply(I1, 255.0)
	I2_ = tf.multiply(I2, 255.0)
	Igt_ = tf.multiply(Igt, 255.0)

	I_ = tf.cast(I_, tf.uint8)
	I1_ = tf.cast(I1_, tf.uint8)
	I2_ = tf.cast(I2_, tf.uint8)
	Igt_ = tf.cast(Igt_, tf.uint8)

	psnr = tf.image.psnr(Igt_[0][:,:,:], I_[0][:,:,:], max_val=255)
	ssim = tf.image.ssim(Igt_[0][:,:,:], I_[0][:,:,:], max_val=255)
	ms_ssim = tf.image.ssim_multiscale(Igt_[0][:,:,:], I_[0][:,:,:], max_val=255)

	saver = tf.train.Saver(tf.global_variables())

	with tf.Session() as sess:
		# Initialize
		sess.run(tf.global_variables_initializer())

		# Restore weights of model
		saver.restore(sess, model_dir)

		# Validation
		log = "\n========== Validation Begin ==========\n"
		write_logs(logs_dir, log, True)
		valid_start = time.time()
		avg_l1 = 0
		avg_lf = 0
		avg_lc = 0
		avg_psnr = 0
		avg_ssim = 0
		avg_ms_ssim = 0
		for path in valid_gt:
			valid_fr_start = time.time()
			F1, F2, Fgt = sess.run([fr1, fr2, gt])
			_, intp_fr, L1_val, Lf_val, Lc_val, psnr_val, ssim_val, ms_ssim_val = sess.run([I, I_, L1, Lf, Lc, psnr, ssim, ms_ssim], feed_dict={I1:F1, I2:F2, Igt:Fgt})

			avg_l1 += L1_val
			avg_lf += Lf_val
			avg_lc += Lc_val
			avg_psnr += psnr_val
			avg_ssim += ssim_val
			avg_ms_ssim += ms_ssim_val

			_, _, _, scene, _, frame = path.split("\\")
			fr_dir = intp_dir + "\\" + scene + "\\" + frame
			img_write(fr_dir, intp_fr, 'PNG-FI')

			valid_fr = scene + '/60fps/' + frame
			log = "Image {}, Time {:2.5f}, Shape = {}, L1 Loss = {:2.5f}, Perceptual Loss = {:2.5f}, Combined Loss = {:2.5f}, PSNR = {:2.5f} dB, SSIM = {:2.5f}, MS-SSIM = {:2.5f}"\
						.format(valid_fr, time.time()-valid_fr_start, intp_fr.shape, L1_val, Lf_val, Lc_val, psnr_val, ssim_val, ms_ssim_val)
			write_logs(logs_dir, log, False)

		log = "\nAverage L1 Loss = {:2.5f}".format(avg_l1/num_valid)
		write_logs(logs_dir, log, False)
		log = "Average Perceptual Loss = {:2.5f}".format(avg_lf/num_valid)
		write_logs(logs_dir, log, False)
		log = "Average Combined Loss = {:2.5f}".format(avg_lc/num_valid)
		write_logs(logs_dir, log, False)
		log = "Average PSNR = {:2.5f} dB".format(avg_psnr/num_valid)
		write_logs(logs_dir, log, False)
		log = "Average SSIM = {:2.5f}".format(avg_ssim/num_valid)
		write_logs(logs_dir, log, False)
		log = "Average MS-SSIM = {:2.5f}\n".format(avg_ms_ssim/num_valid)
		write_logs(logs_dir, log, False)
		log = "\nValidation Time: {:2.5f}".format(time.time()-valid_start)
		write_logs(logs_dir, log, False)
		log = "\n========== Validation End ==========\n"
		write_logs(logs_dir, log, False)
		sess.close()



