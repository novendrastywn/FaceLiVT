from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4) # combine (1.0, 0.0, 0.4) archloss (1.0, 0.5, 0.0)
config.aux_margin_list = "None"
config.network = "facelivtv2_xs"
config.resume = False
config.output = 'work_dirs/facelivtv2/facelivtv2_xs' #baseline
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.weight_decay = 1e-2 #0.025 #1e-4 #0.05
config.batch_size = 1024
config.optimizer = "adamw"
config.lr = 6e-3
config.verbose = 1000
config.frequent = 1000
config.ver_start = 30 #ver_start_epoch
config.dali = False #False

#distilation
config.distillation = 'None'
config.teacher_network = 'facelivt_l'
config.teacher_path = '/home/dsdl/Documents/Face-Recog_Workspace/work_dirs/facelivt_l/model.pt'

config.alpha = 1.0
config.beta  = 0.5
config.gama  = 0.5

# config.dali_aug = True

config.num_workers = 6

config.rec = "/home/ndr/Container/ImageDataset/face_dataset/glint360k"
config.num_classes = 360232
config.num_image = 17091657
config.epoch = 50
config.warmup_epoch = 0
config.val_targets = ['lfw', 'calfw', 'cplfw', 'cfp_fp', 'agedb_30']

config.scheduler = "poly"
config.warmup_lr = 6e-3
config.warmup_epochs = 0 
config.min_lr = 0



