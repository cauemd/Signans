import tensorflow as tf
import tensorflow_hub as hub
import threading
import cv2 
import numpy as np
import sys
import os
import time
import six
import argparse
import uuid
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from gtts import gTTS
from IPython.display import Audio
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from autocorrect import Speller
from pydub import AudioSegment
from pydub.playback import play
from google.cloud import translate_v2 as translate
from random import *

google_translation = os.path.join(os.getcwd(), "stately-vector.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  google_translation

config_file = os.path.join('TensorFlow-Model', 'pipeline.config')
config = config_util.get_configs_from_pipeline_file(config_file)

# Loading the model
detection_model = model_builder.build(model_config=config['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('TensorFlow-Model', 'checkpoint', 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# Setup capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


args = parser.parse_args()
   

lastest_reading = ""
phrase = ""
TGREEN =  '\033[32m'
ENDC = '\033[m'
#Args i.e. "python Signans.py --language fr --speed 1"
parser = argparse.ArgumentParser()
parser.add_argument('--language', "-l")
parser.add_argument('--speed', '-s')
if args.language != None:
    output_language = str(args.language)
else:
    output_language = "en"

if args.speed != None:
    detection_speed = float(args.speed)
else:
    detection_speed =  2

top_domain = {'pt-br':'com.br',
              'pt':'pt',
              'fr':'fr',
              'fr-ca':'ca',
              'en-uk':'co.uk',
              'en':'com',
              'es':'es',
              'es-mx':'com.mx',
              'zh-CN' : 'com'
            }
lang = output_language.split('-')[0]

def play_audio(voice_file):
    sound = AudioSegment.from_mp3(voice_file)
    play(sound)
    os.remove(voice_file)

def get_lastest_reading():
    global phrase
    while get_lastest_reading_thread.is_alive():
        global lastest_reading
        label = lastest_reading
        if label != "":
            if label == "dot":
                if phrase != "":
                    spell = Speller()
                    phrase = spell(phrase)
                    if lang != "en":
                        translate_client = translate.Client()
                        translated_test = translate_client.translate(phrase, target_language=lang)["translatedText"]
                        voice = gTTS( text=translated_test, lang=lang, tld=top_domain[output_language], slow=False)
                    else:
                        voice = gTTS( text=phrase, lang=lang, tld=top_domain[output_language], slow=False) 
                    voice_file = "voice-{}.mp3".format(str(randint(1, 9999999)))         
                    voice.save(voice_file)
                    play_audio_thread = threading.Thread(target=play_audio, args=[voice_file])
                    play_audio_thread.daemon = True
                    play_audio_thread.start()
                    print (TGREEN + "audio:" + phrase, ENDC)
                    phrase = ""                    
            elif label == "space":
                phrase += " "
                spell = Speller()
                phrase = spell(phrase)
                print (TGREEN + phrase, ENDC) 
            else:
                phrase += label
                print (TGREEN + label, ENDC)
            time.sleep(detection_speed)
            lastest_reading = ""

get_lastest_reading_thread = threading.Thread(target=get_lastest_reading)
get_lastest_reading_thread.daemon = True

while True:
    ret, frame = cap.read()
    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()
    # Accuracy of 85%
    if detections['detection_scores'][0] > 0.85:
        index = detections['detection_classes'][0]+1
        label = category_index[index]['name']
        cv2.putText(image_np_with_detections, label.upper() ,(50,150), cv2.FONT_HERSHEY_SIMPLEX, 3,(0, 0, 0),6,cv2.LINE_AA)
        cv2.putText(image_np_with_detections, str(detections['detection_scores'][0]*100)[0:4]+"%" ,(50,250), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 0),6,cv2.LINE_AA)
        if lastest_reading == "":           
            if not get_lastest_reading_thread.is_alive():
                get_lastest_reading_thread.start()
            lastest_reading = label
    cv2.putText(image_np_with_detections,phrase.replace(" ", "-"),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0),3,cv2.LINE_AA)
    cv2.imshow('ASL - Signans',  cv2.resize(image_np_with_detections, (640, 480)))   
    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
    # Press B to delete last letter
    if cv2.waitKey(1) & 0xFF == ord('b'):
        if len(phrase) >= 1:
            phrase = phrase[0: -1]
        else:
            phrase = ""