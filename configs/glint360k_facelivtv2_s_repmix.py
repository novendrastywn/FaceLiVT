from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "facelivt_s_repmix"
config.resume = False
config.output = 'work_dirs/facelivtv2/s_repmix'
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.weight_decay = 1e-4 #0.025 #1e-4 #0.05
config.batch_size = 256
config.optimizer = "adamw"
config.lr = 6e-3
config.verbose = 10000
config.frequent = 1000
config.dali = False #False

#distilation
config.distillation = 'none'
config.teacher_network = 'facelivt_l'
config.teacher_path = '/home/dsdl/Documents/Face-Recog_Workspace/work_dirs/facelivt_l/model.pt'
config.alpha = 0.5

# config.dali_aug = True

config.num_workers = 28

config.rec = "/home/dsdl/Documents/ImageDataset/glint360k/"
config.num_classes = 360232
config.num_image = 17091657
config.num_epoch = 40
config.warmup_epoch = 0
config.val_targets = ['lfw', 'calfw', 'cplfw', 'agedb_30', 'cfp_fp']

