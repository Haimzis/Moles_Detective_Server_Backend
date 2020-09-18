from easydict import EasyDict as edict

MIN_POSSIBLE_PIXELS_FOR_RECOGNITION = 25
### NETS ###
INPUT_SIZE = 250

net_params = edict()
net_params.classification = edict()
net_params.segmentation = edict()

#### Segmentation ####
net_params.segmentation.input_tensor_name = 'ImageTensor:0'
net_params.segmentation.output_tensor_name = 'SemanticPredictions:0'
net_params.segmentation.frozen_model_name = 'MobileNet_V3'
net_params.segmentation.frozen_model = '../files/Models/MobileNet_V3_large_MolesDetective_ver1.pb'
net_params.segmentation.batch_size = 1

#### Classification ####
net_params.classification.input_tensor_name = 'fifo_queue_Dequeue:0'
net_params.classification.output_tensor_name = 'InceptionV3/Predictions/Reshape_1:0'
net_params.classification.frozen_model_name = 'Inception_V3'
net_params.classification.frozen_model = '../files/Models/InceptionV3_ISIC_ver1.pb'
net_params.classification.batch_size = 12
