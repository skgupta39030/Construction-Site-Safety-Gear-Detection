import streamlit as st
from db import Video, Base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os
import cv2 as cv
import tempfile
import pickle
import numpy as np

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


if os.path.exists('db.sqlite3'):
    engine = create_engine('sqlite:///db.sqlite3')
    Base.metadata.create_all(engine)
    st.sidebar.success("database created")
else:
    st.sidebar.success("database loaded")

# model related setup
MODELS_DIR = os.path.join(os.getcwd(), 'models')
MODEL_NAME = 'my_ssd_resnet50_v1_fpn'
PATH_TO_CKPT = os.path.join(MODELS_DIR,MODEL_NAME )
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
LABEL_FILENAME = 'label_map.pbtxt'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))

def load_model(checkpoint='ckpt-17'):
    # st.sidebar.text(f'path to ckpt {PATH_TO_CKPT}')
    # st.sidebar.text(f'path to cfg {PATH_TO_CFG}')
    st.sidebar.info(f"path exists {os.path.exists(PATH_TO_CKPT)}")
    if not os.path.exists(PATH_TO_CKPT):
        st.error('Checkpoint is unvalid')
    try:
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
        model_config = configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(PATH_TO_CKPT, checkpoint)).expect_partial()
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        return detection_model,category_index
    except Exception as e:
        st.sidebar.error(e)


@tf.function
def detect_fn(image,detection_model):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections, prediction_dict, tf.reshape(shapes, [-1])

model,category_index = load_model('ckpt-17')
if model:
    st.sidebar.success("model loaded successfully")
    st.sidebar.subheader("detectable labels in model")
    st.sidebar.write(category_index)

st.title("Construction Site Saftey Gear Detection")
st.subheader('by Sudhanshu')


def opendb():
    engine = create_engine('sqlite:///db.sqlite3')  # connect
    Session = sessionmaker(bind=engine)
    return Session()


def save_file(path):
    try:
        db = opendb()
        file = os.path.basename(path)
        name, ext = file.split('.')  # second piece
        vid = Video(filename=name, extension=ext, filepath=path)
        db.add(vid)
        db.commit()
        db.close()
        return True
    except Exception as e:
        st.write("database error:", e)
        return False

form = st.form(key="vid form")
name_of_video = form.text_input('video name (without ext)', value='example_video')
f = form.file_uploader("upload a file")
btn = form.form_submit_button('save')

if f and btn:
    folder = 'data'
    path = os.path.join(folder, f"{name_of_video}.mp4")
    save_file(path)
    file_details = {"FileName": f.name, "FileType": f.type}
    st.write(file_details)

    with open(path, "wb") as fs:
        fs.write(f.getbuffer())
        st.success("Saved File")

db = opendb()
videos = db.query(Video).all()
db.close()
vid = st.selectbox('select a video to play', videos)
if vid and os.path.exists(vid.filepath):
    st.video(vid.filepath)
    threshold = st.slider("minimun detection threshold",min_value=0.10, max_value=0.90, value=0.30, step=0.05)
    btn =  st.button("start AI based detection")

    if btn:
        cap = cv.VideoCapture(vid.filepath)
        st.info("please wait for the window to launch, to close the popup window, press 'q'.")
        while True:
            ret, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections, predictions_dict, shapes = detect_fn(input_tensor,model)
            label_id_offset = 1
            image_np_with_detections = image_np.copy()
            viz_utils.visualize_boxes_and_labels_on_image_array( image_np_with_detections,  detections['detection_boxes'][0].numpy(),
                                                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                                                detections['detection_scores'][0].numpy(), category_index,
                                                use_normalized_coordinates=True, max_boxes_to_draw=10,
                                                min_score_thresh=threshold, agnostic_mode=False)
            # Display output
            cv.imshow('object detection', cv.resize(image_np_with_detections, (800, 600)))

            if cv.waitKey(25) & 0xFF == ord('q'):
                st.warning("you stopped the video")
                break
        

        cap.release()
        cv.destroyAllWindows()
