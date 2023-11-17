import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import torch
import cv2
class issbaEncoder(object):
    def __init__(self,model_path, secret,size) :
        BCH_POLYNOMIAL = 137
        BCH_BITS = 5
        self.size = size
        
        self.sess = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        model = tf.compat.v1.saved_model.loader.load(self.sess, [tag_constants.SERVING], model_path)
        input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
        input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
        self.input_secret = tf.compat.v1.get_default_graph().get_tensor_by_name(input_secret_name)
        self.input_image = tf.compat.v1.get_default_graph().get_tensor_by_name(input_image_name)
        output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
        output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
        self.output_stegastamp = tf.compat.v1.get_default_graph().get_tensor_by_name(output_stegastamp_name)
        self.output_residual = tf.compat.v1.get_default_graph().get_tensor_by_name(output_residual_name)
        
        
        
        bch = bchlib.BCH(t=BCH_BITS,prim_poly=BCH_POLYNOMIAL)
        if len(secret) > 7:
            print('Error: Can only encode 56bits (7 characters) with ECC')
            return
        data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')
        ecc = bch.encode(data)
        packet = data + ecc
        packet_binary = ''.join(format(x, '08b') for x in packet)
        secret = [int(x) for x in packet_binary]
        secret.extend([0,0,0,0])
        self.secret = secret
    @tf.function  
    def compute_output(self, input_image, input_secret):
        return self.output_stegastamp, self.output_residual
    def __call__(self, image):
        input_data = np.array(image, dtype=np.float32)
        input_data = np.transpose(input_data, (1, 2, 0))
        
        
        hidden_img, _ = self.compute_output(input_image=input_data, input_secret=self.secret)
        
        output = hidden_img[0].numpy()  
        output = torch.tensor(np.transpose(output, (2, 0, 1)))
        return output
    
    
    
    
    
    
    
    
    
    
    
    