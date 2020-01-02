from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()
config.VALID = edict()
config.IMG = edict()
config.TEST = edict()

# Image Parameters
config.IMG.height = 720
config.IMG.width = 1280

# Hyper Parameters
# Training
config.TRAIN.num_epoches = 50
config.TRAIN.patch_size = 128
config.TRAIN.batch_size = 16
config.TRAIN.learning_rate_init = 1e-3
config.TRAIN.learning_rate_decay = 0.5
config.TRAIN.decay_period = 10
config.TRAIN.gamma = 1e-4

config.TRAIN.model_dir = '..\\model\\model.ckpt'
config.TRAIN.logs_dir = '..\\logs'
config.TRAIN.logs_train = '..\\logs\\logs_train.txt'

# Validation
config.VALID.intp_dir = '..\\report\\valid_intp'
config.VALID.logs_valid = '..\\logs\\logs_valid.txt'

# Test
config.TEST.datapath = '..\\data\\test'
config.TEST.logs_test = '..\\logs\\logs_test.txt'
config.TEST.result_path = '..\\report'
config.TEST.intp_dir = '..\\report\\test_intp'
