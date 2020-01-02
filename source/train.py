import tensorflow as tf
import numpy as np
import os
import time
import csv

import vgg19

from config import config
from model import ASCNet
from utils import write_logs
from valid import valid_parse

# Directories
model_dir = config.TRAIN.model_dir
logs_dir = config.TRAIN.logs_dir
logs_train = config.TRAIN.logs_train

# Parameters
num_epoches = config.TRAIN.num_epoches
patch_size = config.TRAIN.patch_size
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.learning_rate_init
lr_decay = config.TRAIN.learning_rate_decay
lr_decay_period = config.TRAIN.decay_period
gamma = config.TRAIN.gamma

# Image
height = config.IMG.height
width = config.IMG.width

def train_datapath():
  list_fr1 = []
  list_fr2 = []
  list_gt = []
  for v in range(1,6):
    with open('..\\data\\train\\video%1d.csv'%v) as video:
      reader = csv.DictReader(video)
      s = 1
      for row in reader:
        scene_begin = int(row['scene_begin'])
        scene_end = int(row['scene_end'])
        for fr in range(scene_begin, scene_end-1):
          fr1_path = "..\\data\\train\\video{}\\scene_{:03d}\\{:06d}.png".format(v, s, fr)
          fr2_path = "..\\data\\train\\video{}\\scene_{:03d}\\{:06d}.png".format(v, s, fr+2)
          gt_path = "..\\data\\train\\video{}\\scene_{:03d}\\{:06d}.png".format(v, s, fr+1)
          list_fr1.append(fr1_path)
          list_fr2.append(fr2_path)
          list_gt.append(gt_path)
        s = s + 1
      video.close()
  return list_fr1, list_fr2, list_gt

def train_parse(fr1_dir, fr2_dir, gt_dir):
	fr1_str = tf.read_file(fr1_dir)
	fr2_str = tf.read_file(fr2_dir)
	gt_str = tf.read_file(gt_dir)

	fr1 = tf.image.decode_png(fr1_str, channels=3)
	fr2 = tf.image.decode_png(fr2_str, channels=3)
	gt = tf.image.decode_png(gt_str, channels=3)

	images = tf.concat([fr1, fr2, gt], axis=2)
	return images

def train_preprocess(images):
	patches = tf.random_crop(images, [patch_size, patch_size, 9])
	patches = tf.image.random_flip_left_right(patches)
	patches = tf.image.random_flip_up_down(patches)

	fr1, fr2, gt = tf.split(patches, 3, axis=2)

	fr1 = tf.image.convert_image_dtype(fr1, tf.float32)
	fr2 = tf.image.convert_image_dtype(fr2, tf.float32)
	gt = tf.image.convert_image_dtype(gt, tf.float32)

	return fr1, fr2, gt

def training():
	I1 = tf.placeholder(tf.float32, [None, patch_size, patch_size, 3], name='Patches1')
	I2 = tf.placeholder(tf.float32, [None, patch_size, patch_size, 3], name='Patches2')
	Igt = tf.placeholder(tf.float32, [None, patch_size, patch_size, 3], name='PatchesGT')
	with tf.name_scope('ASCNet') as ascnet:
		I = ASCNet(I1, I2, is_train=True, reuse=False)
	I_ = tf.clip_by_value(I, 0.0, 1.0)
	
	train_fr1, train_fr2, train_gt = train_datapath()
	num_train = len(train_fr1)
	train_data = tf.data.Dataset.from_tensor_slices((train_fr1, train_fr2, train_gt))
	train_data = train_data.shuffle(num_train)
	train_data = train_data.map(train_parse, num_parallel_calls=4)
	train_data = train_data.map(train_preprocess, num_parallel_calls=4)
	train_data = train_data.batch(batch_size)
	train_iter = train_data.make_initializable_iterator()
	fr1, fr2, gt = train_iter.get_next()
	
	valid_fr1 = ["..\\data\\eval\\1\\eval_00738.png", "..\\data\\eval\\2\\eval_01638.png", "..\\data\\eval\\3\\eval_03381.png"]
	valid_fr2 = ["..\\data\\eval\\1\\eval_00740.png", "..\\data\\eval\\2\\eval_01640.png", "..\\data\\eval\\3\\eval_03383.png"]
	valid_gt = ["..\\data\\eval\\1\\eval_00739.png", "..\\data\\eval\\2\\eval_01639.png", "..\\data\\eval\\3\\eval_03382.png"]
	valid_data = tf.data.Dataset.from_tensor_slices((valid_fr1, valid_fr2, valid_gt))
	valid_data = valid_data.map(valid_parse, num_parallel_calls=4)
	valid_data = valid_data.batch(3)
	valid_iter = valid_data.make_initializable_iterator()
	vfr1, vfr2, vgt = valid_iter.get_next()
	
	# Loss Functions
	with tf.name_scope('L1_Loss'):
		L1 = tf.losses.absolute_difference(Igt, I)
	sum_l1_op = tf.summary.scalar("L1_Loss", L1)

	vgg19_gt = vgg19.Vgg19()
	vgg19_gt.build(Igt)
	vgg19_intp = vgg19.Vgg19()
	vgg19_intp.build(I_)
	with tf.name_scope('Perceptual_Loss'):
		Lf = tf.losses.mean_squared_error(vgg19_gt.conv4_4, vgg19_intp.conv4_4)
	sum_lf_op = tf.summary.scalar("Perceptual_Loss", Lf)
	
	with tf.name_scope('Combined_Loss'):
		Lc = L1 + gamma*Lf
	sum_lc_op = tf.summary.scalar("Combined_Loss", Lc)
	
	sum_loss_op = tf.summary.merge([sum_l1_op, sum_lf_op, sum_lc_op])

	# Learning Rate
	with tf.variable_scope('learning_rate'):
		lr_v = tf.Variable(lr_init, trainable=False)

	# Optimizer
	optimizer = tf.train.AdamOptimizer(lr_v)
	gvs = optimizer.compute_gradients(Lc)
	capped_gvs = [(tf.clip_by_value(grad,-0.5, 0.5), var) for grad, var in gvs]
	train_op = optimizer.apply_gradients(capped_gvs)

	saver = tf.train.Saver()

	I_ = tf.multiply(I, 255.0)
	I1_ = tf.multiply(I1, 255.0)
	I2_ = tf.multiply(I2, 255.0)
	Igt_ = tf.multiply(Igt, 255.0)

	I_ = tf.cast(I_, tf.uint8)
	I1_ = tf.cast(I1_, tf.uint8)
	I2_ = tf.cast(I2_, tf.uint8)
	Igt_ = tf.cast(Igt_, tf.uint8)

	sum_fr1_op = tf.summary.image("Frame_1", I1_)
	sum_fr2_op = tf.summary.image("Frame_2", I2_)
	sum_gt_op = tf.summary.image("Ground_Truth", Igt_)
	sum_intp_op = tf.summary.image("Interpolation_Frame", I_)
	sum_fr_op = tf.summary.merge([sum_fr1_op, sum_fr2_op, sum_gt_op])

	psnr0 = tf.image.psnr(Igt_[0][:,:,:], I_[0][:,:,:], max_val=255)
	psnr1 = tf.image.psnr(Igt_[1][:,:,:], I_[1][:,:,:], max_val=255)
	psnr2 = tf.image.psnr(Igt_[2][:,:,:], I_[2][:,:,:], max_val=255)
	sum_psnr_op0 = tf.summary.scalar("PSNR0", psnr0)
	sum_psnr_op1 = tf.summary.scalar("PSNR1", psnr1)
	sum_psnr_op2 = tf.summary.scalar("PSNR2", psnr2)
	sum_psnr_op = tf.summary.merge([sum_psnr_op0, sum_psnr_op1, sum_psnr_op2])

	ssim0 = tf.image.ssim(Igt_[0][:,:,:], I_[0][:,:,:], max_val=255)
	ssim1 = tf.image.ssim(Igt_[1][:,:,:], I_[1][:,:,:], max_val=255)
	ssim2 = tf.image.ssim(Igt_[2][:,:,:], I_[2][:,:,:], max_val=255)
	sum_ssim_op0 = tf.summary.scalar("SSIM0", ssim0)
	sum_ssim_op1 = tf.summary.scalar("SSIM1", ssim1)
	sum_ssim_op2 = tf.summary.scalar("SSIM2", ssim2)
	sum_ssim_op = tf.summary.merge([sum_ssim_op0, sum_ssim_op1, sum_ssim_op2])
	
	if num_train % batch_size != 0:
		num_batches = int(num_train/batch_size) + 1
	else:
		num_batches = int(num_train/batch_size)

	with tf.Session() as sess:
		# Initialize variables
		sess.run(tf.global_variables_initializer())

		# Op to write logs to Tensorboard
		train_sum_writer = tf.summary.FileWriter(logs_dir, tf.get_default_graph())

		# Training Process
		log = "\n========== Training Begin ==========\n"
		write_logs(logs_train, log, True)
		train_start = time.time()
		for epoch in range(num_epoches):
			epoch_start = time.time()
			
			if (epoch > 19) and (epoch % lr_decay_period == 0):
				new_lr = lr_v * lr_decay
				sess.run(tf.assign(lr_v, new_lr))
				log = "** New learning rate: %1.9f **\n" % (lr_v.eval())
				write_logs(logs_train, log, False)
			elif epoch == 0:
				sess.run(tf.assign(lr_v, lr_init))
				log = "** Initial learning rate: %1.9f **\n" % (lr_init)
				write_logs(logs_train, log, False)
			
			avg_l1 = 0
			avg_lf = 0
			avg_lc = 0
			sess.run(train_iter.initializer)
			for batch in range(num_batches):
				batch_start = time.time()
				F1, F2, Fgt = sess.run([fr1, fr2, gt])
				_, L1_val, Lf_val, Lc_val, sum_loss = sess.run([train_op, L1, Lf, Lc, sum_loss_op], feed_dict={I1:F1, I2:F2, Igt:Fgt})
				avg_l1 += L1_val
				avg_lf += Lf_val
				avg_lc += Lc_val
				train_sum_writer.add_summary(sum_loss, epoch*num_batches+batch)
				log = "Epoch {}, Time {:2.5f}, Batch {}, L1 Loss = {:2.5f}, Perceptual Loss = {:2.5f}, Combined Loss = {:2.5f}"\
							.format(epoch, time.time()-batch_start, batch, L1_val, Lf_val, Lc_val)
				write_logs(logs_train, log, False)
			log = "\nEpoch {}, Time {:2.5f}, Average L1 Loss = {:2.5f}, Average Perceptual Loss = {:2.5f}, Average Combined Loss = {:2.5f}\n"\
						.format(epoch, time.time()-epoch_start, avg_l1/num_batches, avg_lf/num_batches, avg_lc/num_batches)
			write_logs(logs_train, log, False)
			
			sess.run(valid_iter.initializer)
			Vf1, Vf2, Vgt = sess.run([vfr1, vfr2, vgt])
			valid_dict={I1:Vf1, I2:Vf2, Igt:Vgt}
			if epoch == 0:
				_, sum_fr = sess.run([I, sum_fr_op], feed_dict=valid_dict)
				train_sum_writer.add_summary(sum_fr, epoch)
			_, sum_intp, sum_psnr, sum_ssim = sess.run([I, sum_intp_op, sum_psnr_op, sum_ssim_op], feed_dict=valid_dict)
			train_sum_writer.add_summary(sum_intp, epoch)
			train_sum_writer.add_summary(sum_psnr, epoch)
			train_sum_writer.add_summary(sum_ssim, epoch)
			#train_sum_writer.add_summary(sum_ms_ssim, epoch)
			
		log = "\nTraining Time: {}".format(time.time()-train_start)
		write_logs(logs_train, log, False)
		log = "\n========== Training End ==========\n"
		write_logs(logs_train, log, False)

		# Save model
		save_path = saver.save(sess, model_dir)
		log = "Model is saved in file: %s" % save_path
		write_logs(logs_train, log, False)
		sess.close()


