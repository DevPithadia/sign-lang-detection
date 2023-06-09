#!/usr/bin/env python
# coding: utf-8

# Resources Used
# - wget.download('https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py')
# - Setup https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

# # 0. Setup Paths

# In[24]:


WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
import pandas as pd
import pickle
import streamlit as st
from IPython import get_ipython
import subprocess

#from streamlit_option_menu import option_menu


# # 1. Create Label Map

# In[25]:


labels = [{'name':'yes', 'id':1}, {'name':'hello', 'id':2}, {'name':'thank you', 'id':3}]

with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')


# # 2. Create TF records

# In[26]:

scripts_path = "/path/to/scripts"
image_path = "/path/to/images"
annotation_path = "/path/to/annotations"

# Generate train.record
train_command = f"python {scripts_path}/generate_tfrecord.py -x {image_path}/train -l {annotation_path}/label_map.pbtxt -o {annotation_path}/train.record"
subprocess.run(train_command, shell=True)

# Generate test.record
test_command = f"python {scripts_path}/generate_tfrecord.py -x {image_path}/test -l {annotation_path}/label_map.pbtxt -o {annotation_path}/test.record"
subprocess.run(test_command, shell=True)


# # 3. Download TF Models Pretrained Models from Tensorflow Model Zoo

# In[27]:


get_ipython().system('cd Tensorflow && git clone https://github.com/tensorflow/models')


# In[28]:


#wget.download('http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz')
#!mv ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz {PRETRAINED_MODEL_PATH}
#!cd {PRETRAINED_MODEL_PATH} && tar -zxvf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz


# # 4. Copy Model Config to Training Folder

# In[29]:


CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 


# In[30]:


get_ipython().system("mkdir {'Tensorflow\\\\workspace\\\\models\\\\'+CUSTOM_MODEL_NAME}")
get_ipython().system("cp {PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config'} {MODEL_PATH+'/'+CUSTOM_MODEL_NAME}")


# # 5. Update Config For Transfer Learning

# In[31]:


import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


# In[32]:


CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'


# In[33]:


config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)


# In[34]:


config


# In[35]:


pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)  


# In[36]:


pipeline_config.model.ssd.num_classes = 3
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']


# In[37]:


config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)   


# # 6. Train the model

# In[38]:


print("""python {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={}/{}/pipeline.config --num_train_steps=20000""".format(APIMODEL_PATH, MODEL_PATH,CUSTOM_MODEL_NAME,MODEL_PATH,CUSTOM_MODEL_NAME))


# In[39]:


# from keras.models import load
# Load pipeline config and build a detection model
# configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
# detection_model = model_builder.build(model_config=configs['model'], is_training=False)
# detection_model.save('signLanguage.h5')


# # 7. Load Train Model From Checkpoint

# In[40]:


import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


# In[41]:


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-22')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# In[42]:


##Saving the model
filename = "sl_model.sav"
pickle.dump(CUSTOM_MODEL_NAME,open(filename,'wb'))


# In[43]:


sign_model = pickle.load(open('sl_model.sav','rb'))


# In[ ]:





# # 8. Detect in Real-Time

# In[44]:


import cv2 
import numpy as np


# In[45]:


category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')


# In[ ]:





# In[46]:


# Setup capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# In[ ]:


while True: 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break


# In[ ]:


detections = detect_fn(input_tensor)


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


cap.release()


# In[ ]:




