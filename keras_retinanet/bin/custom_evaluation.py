import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.preprocessing.csv_generator import _read_classes

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import csv
from glob import glob
import pickle

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def evaluation_run(model_path, image_folder, img_extention, category_csv, output_folder, threshold, save_flag):

    # keras session
    keras.backend.tensorflow_backend.set_session(get_session())

    # load model
    model = models.load_model(model_path, backbone_name='resnet50')

    # load categories
    with open(category_csv) as file:
        classes = _read_classes(csv.reader(file, delimiter=','))
    class_mapping = {v: k for k, v in list(classes.items())}

    # load image names
    images = glob(image_folder + '/*' + img_extention)

    # run evaluation
    for image_path in images:

        # load image
        image = read_image_bgr(image_path)

        # intialize output annotations
        output = {'img_name':os.path.split(image_path)[-1],
                  'bboxes':[]}

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < threshold:
                break

            color = label_color(label)

            b = box.astype(int)

            x1 = b[0]
            y1 = b[1]
            x2 = b[2]
            y2 = b[3]

            if save_flag == 'image':
                draw_box(draw, b, color=color)
                caption = "{} {:.3f}".format(class_mapping[label], score)
                draw_caption(draw, b, caption)

            elif save_flag == 'annotation':
                output['bboxes'].append({'x1':x1,
                                         'x2':x2,
                                         'y1':y1,
                                         'y2':y2,
                                         'category':class_mapping[label],
                                         'cls_score':score}
                                         )
        if save_flag == 'image':
            plt.imsave(output_folder + os.path.split(image_path)[-1].split('.')[0] + '.png', draw)
        elif save_flag == 'annotation':
            json_file_name = os.path.split(image_path)[-1].split('.')[0] + '.pickle'
            with open(os.path.join(output_folder, json_file_name), 'wb') as handle:
                pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    model_path = '/home/bo_huang/continental_detection/model_best.h5'
    image_folder = '/home/bo_huang/parking_detection'
    img_extention = '.png'
    category_csv = '/home/bo_huang/continental_detection/csv_files/class_mapping.csv'
    output_folder = '/home/bo_huang/parking_detection/evaluation_output'
    threshold = 0.1
    save_flag = 'annotation'
    evaluation_run(model_path, image_folder, img_extention, category_csv, output_folder, threshold, save_flag)