import tensorflow as tf
from io import BytesIO
from urllib import request
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def getmodel():    
    lite_model = tf.lite.Interpreter('bees-wasps-v2.tflite')
    lite_model.allocate_tensors()
    return lite_model

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess(x):
    x /=127.5
    x -=1.
    return x


def main():
    img = download_image('https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg')
    prep_img = prepare_image(img,[150,150])
    num_img = np.array(prep_img,dtype='float32')
    preprocess(num_img)
    train_gen = ImageDataGenerator()
    pre_img = train_gen.apply_transform(num_img,transform_parameters={
    'theta':50,
    'tx':0.1,
    'ty':0.1,
    'zx':0.1,
    'zy':0.1,
    'flip_horizontal':True 
    })

    lite_model = getmodel()
    input_indx = lite_model.get_input_details()[0]['index']
    output_index = lite_model.get_output_details()[0]['index']

    pre_img = np.array([pre_img])


    lite_model.set_tensor(input_indx,pre_img)
    lite_model.invoke()
    prep = lite_model.get_tensor(output_index)
    print(prep[0][0])

if __name__ == "__main__":
    main()