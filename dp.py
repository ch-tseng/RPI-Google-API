import numpy as np
import os
import datetime
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
import io, time, sys
from matplotlib import pyplot as plt
from PIL import Image
import picamera
camera = picamera.PiCamera()

sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

#----my library-------------#
from libraryCH.device.lcd import ILI9341
lcd = ILI9341(LCD_size_w=240, LCD_size_h=320, LCD_Rotate=270)

#----Configuration----------#
# ssd_mobilenet_v1_coco_11_06_2017 / ssd_inception_v2_coco_11_06_2017
MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
IMAGE_SIZE = (6.4, 4.8)
BoxLineThickness = 5

camera.rotation = 0
#camera.resolution = (1296, 972)
camera.resolution = (640, 480)
#camera.hflip = True
#camera.vflip = True


#-----Don't need to touch -------------#

# Functions--------------------
#Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def PIL2array(img):
  return np.array(img.getdata(),
              np.uint8).reshape(img.size[1], img.size[0], 3)

def array2PIL(arr, size):
  mode = 'RGBA'
  arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
  if len(arr[0]) == 3:
    arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

def resizeIMG(img, width=320, hsize=240):
  #wpercent = (width/float(img.size[0]))
  #hsize = int((float(img.size[1])*float(wpercent)))
  img = img.resize((width,hsize), Image.ANTIALIAS)
  return img

def displayIMG(img, pltshow=0, lcdDisplay=1, savePic=0):
  if(savePic==1):
    now = "{:%Y-%m-%d_%H%M}".format( datetime.datetime.now() )
    filename = "/home/pi/pics/" + now + ".jpg"
    im = Image.fromarray(img)
    im.save(filename)

  if(lcdDisplay==1):
    displayIMG = Image.fromarray(img)
    displayIMG = resizeIMG(displayIMG, 320, 320)
    displayIMG = displayIMG.rotate(180)

    x, y = displayIMG.size
    lcdImage = Image.new("RGB", (320,320), "white")
    lcdImage.paste(displayIMG, (0,0,x,y))
    lcd.displayImg(np.array(displayIMG) )

  if(pltshow==1):
    plt.close('all')
    #plt.axis("off")
    #plt.ion() # ---> Interactive mode on, this will wait for the plt window be closed.
    plt.ioff()  # ---> Interactive mode off, this will not wait for the plt window be closed.
    plt.imshow(image_np)
    plt.show()


categoriesDetected = ()
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

#Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

ii = 0
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    while True:
      #camera.capture("cap.jpg")
      stream = io.BytesIO()
      camera.capture(stream, format='jpeg')
      stream.seek(0)
      image = Image.open(stream)
      #image = Image.open("cap.jpg")
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=BoxLineThickness)

      #displayIMG(image_np, pltshow=0, lcdDisplay=1, savePic=1)

      ii += 1
      print( "Picture #{}".format(ii))
      print( vis_util.categoriesDetected)

      numPeople = 0
      for object in vis_util.categoriesDetected:
        if(object.find("person")>=0):
          numPeople += 1

      print( "Total person: {} found".format(numPeople))
      savePic=1 if(numPeople>0) else 0

      displayIMG(image_np, pltshow=0, lcdDisplay=1, savePic)
