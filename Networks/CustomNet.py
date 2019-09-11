import os
import colorsys
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import Input
from tensorflow.python.keras.utils import plot_model

from PIL import Image, ImageDraw, ImageFont
from timeit import default_timer as timer
from Networks.TinyYOLOv3 import yolo_body, tiny_yolo_body, yolo_eval
from Networks.mobilenet_v2_yolo3 import mobilenetv2_yolo_body
from DetHelper.utilDet import letterbox_image, get_anchors, get_classes

backbones = ['yolo', 'tiny-yolo', 'mobile']


class YOLOV3():
    def __init__(self, input_shape, load_model=0):
        ''' input_shape (N, N)
            anchors [N x 2]
            classes_names [N,]
        '''
        self.backbone = backbones[load_model]
        self.classes_path = 'YoloFiles/classes.txt' if self.backbone == 'mobile' else 'YoloFiles/coco_classes.txt'
        self.anchors_path = 'YoloFiles/anchors-tiny.txt' if self.backbone == 'tiny-yolo' else 'YoloFiles/anchors.txt'
        self.model_path = \
            ['models/yolo_weights.h5', 'models/yolo-tiny-weights.h5', 'logs/000/train/trained_weights_stage_1.h5'][
                load_model]
        self.input_shape = input_shape
        self.anchors = get_anchors(self.anchors_path)
        self.classes_names = get_classes(self.classes_path)
        self.num_classes = len(self.classes_names)
        self.num_anchors = len(self.anchors) // 3
        self.sess = K.get_session()

        self.score = 0.30
        self.iou = 0.45
        self.gpu_num = 1

        # self.yolo_model.summary()
        self.boxes, self.scores, self.classes = self.generate()

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.classes_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            if self.backbone == 'tiny-yolo':
                self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes)
            elif self.backbone == 'yolo':
                self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            else:
                self.yolo_model = mobilenetv2_yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
                # plot_model(self.yolo_model,'mobilenet_yolo.png',show_shapes=True)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.classes_names), 1., 1.)
                      for x in range(len(self.classes_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.classes_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.input_shape != (None, None):
            assert self.input_shape[0] % 32 == 0, 'Multiples of 32 required'
            assert self.input_shape[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.input_shape)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='DetHelper/font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.classes_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()
