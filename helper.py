import os
import PIL
from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
import tempfile
from pathlib import Path
import uuid
import av




import settings


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def process_image(conf, speciesModel: YOLO, snakeModel:YOLO):
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                # default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            # default_detected_image = PIL.Image.open(
            #     default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):

                res = speciesModel.track(uploaded_image, conf=conf, imgsz=512)
                # Transfer to snake model if no species detected
                if len(res[0].boxes) == 0:
                    res = snakeModel.track(uploaded_image, conf=conf, imgsz=512)

                classnames : dict = res[0].names
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
    
    try: 
        log_results(boxes, classnames)
    except Exception as ex:
        st.write("No detection yet! Make sure to click the button to detect objects first.")


def get_string_results(boxes, classnames, frame = None):
    results = ""
    for box in boxes:
        location = {
            "x1": box.xyxy.numpy()[0].tolist()[0],
            "y1": box.xyxy.numpy()[0].tolist()[1],
            "x2": box.xyxy.numpy()[0].tolist()[2],
            "y2": box.xyxy.numpy()[0].tolist()[3],
        }
        if frame is not None:
            data = {
            "confidence": float(box.conf.numpy()[0]),
            "class": classnames.get(box.cls.numpy()[0]),
            "location": location,
            "frame": frame
        }
        else:
            data = {
                "confidence": float(box.conf.numpy()[0]),
                "class": classnames.get(box.cls.numpy()[0]),
                "location": location,
            }
        results += str(data) + "\n"
    return results

def log_results(boxes, classnames):
        st.write("Detection Results:")
        results = ""
        for box in boxes:
                location = {
                    "x1": box.xyxy.numpy()[0].tolist()[0],
                    "y1": box.xyxy.numpy()[0].tolist()[1],
                    "x2": box.xyxy.numpy()[0].tolist()[2],
                    "y2": box.xyxy.numpy()[0].tolist()[3],
                }
                data = {
                    "confidence": float(box.conf.numpy()[0]),
                    "class": classnames.get(box.cls.numpy()[0]),
                    "location": location,
                }
                results += str(data) + "\n"
        st.code(results, language="json")


def display_tracker_options():
    tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
    return True, tracker_type


def _display_detected_frames(conf, speciesModel: YOLO, snakeModel:YOLO, st_frame, image, expander,  is_display_tracking=None, tracker=None, log_results=False):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    # image = cv2.resize(image, (512, int(512*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = speciesModel.track(image, conf=conf, persist=True, tracker=tracker, imgsz=512)
        if len(res[0].boxes) == 0:
            res = snakeModel.track(image, conf=conf, persist=True, tracker=tracker, imgsz=512)

    else:
        # Predict the objects in the image using the YOLOv8 model
        res = speciesModel.predict(image, conf=conf, imgsz=512)
        if len(res[0].boxes) == 0:
            res = snakeModel.predict(image, conf=conf, imgsz=512)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    if log_results:
        try:
            for box in res[0].boxes:
                data = {
                    "confidence": float(box.conf.numpy()[0]),
                    "class": res[0].names.get(box.cls.numpy()[0]),
                    "location": {
                        "x1": box.xyxy.numpy()[0].tolist()[0],
                        "y1": box.xyxy.numpy()[0].tolist()[1],
                        "x2": box.xyxy.numpy()[0].tolist()[2],
                        "y2": box.xyxy.numpy()[0].tolist()[3],
                    },
                }
                expander.write(data)
        except Exception as ex:
            # st.write(ex)
            expander.write("Video Error!")


def play_youtube_video(conf, speciesModel: YOLO, snakeModel:YOLO):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """

    source_youtube = st.sidebar.text_input("YouTube Video url")
    counter_max = st.slider("Inference Speed (Higher speed = Lower FPS)", 1, 100, 1)

    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            ext = ".webm"
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)
            results_display = st.expander("Detection Results")

            file_id = uuid.uuid4().hex
            fps = vid_cap.get(cv2.CAP_PROP_FPS) / counter_max
            frame_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec = cv2.VideoWriter_fourcc(*'VP90')
            vid_writer = cv2.VideoWriter(f'{file_id}{ext}',  
                                codec, 
                                fps, (frame_width, frame_height)) 
            frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            counter = 0
            log = ""
            expander = st.expander("Detection Results")
            code = expander.code(log, language="json")
            st_frame = st.empty()
            progress_bar = st.progress(0, "Detecing Objects...")
            while (vid_cap.isOpened()):
                counter += 1
                success, image = vid_cap.read()
                if success:
                    if counter % counter_max == 0:
                        results = speciesModel.track(image, conf=conf, imgsz=512, persist=True, tracker=tracker, )
                        if len(results[0].boxes) == 0:
                            results = snakeModel.track(image, conf=conf, imgsz=512, persist=True, tracker=tracker)
                        res_plotted = results[0].plot()
                        st_frame.image(res_plotted,
                                    caption='Detected Video',
                                    channels="BGR",
                                    use_column_width=True
                                    )
                        vid_writer.write(res_plotted)
                        log += get_string_results(results[0].boxes, results[0].names, frame=vid_cap.get(cv2.CAP_PROP_POS_FRAMES))
                        with expander:
                            code.code(log, language="json")
                        progress_bar.progress(int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES)/frames*100), "Detecing Objects...")
                        counter = 0
                else:
                    vid_cap.release()
                    vid_writer.release()
                    break
                
            saved_file_path = Path(f'{file_id}{ext}')
            st.sidebar.success(f"Prediction Completed!")
            with open(saved_file_path, "rb") as file:
                video_bytes = file.read()
            st.video(video_bytes)
            progress_bar.progress(100, "Done!")
            os.remove(saved_file_path)

        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_webcam(conf, snakeModel, speciesModel):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    st.warning("#### ⚠️ Please read the warning below before proceeding.")
    with st.expander("Note: Webcam will not work on deployed app. Please run the app locally to use the webcam. See instructions by exapnding this"):
        st.warning("When using a webcam for live inference, the result will be inaccurate predictions because of the need for fast inference, to keep this realtime. If you want accurate inference we recommend using a video file instead.")
        st.warning("If you are running this app locally, please ignore this message.")
        st.info("To run the app locally, please follow the instructions in the README.md file.")
        st.write("Github Repo:") 
        st.code("https://github.com/khush2003/snake-species-detection-web")
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    results = st.expander("Detection Results")
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             speciesModel,
                                             snakeModel,
                                             st_frame,
                                             image,
                                             results,
                                             is_display_tracker,
                                             tracker,
                                             log_results=True
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.file_uploader(
        "Choose an image...", type=('asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'))
    is_display_tracker, tracker = display_tracker_options()

    # video_bytes = source_vid.read()
    # if video_bytes:
    #     st.video(video_bytes)
    counter_max = st.slider("Inference Speed (Higher speed = Lower FPS)", 1, 10, 1)
    if source_vid is not None:
        # Save temp file with .mp4 extension 
        _, ext = os.path.splitext(source_vid.name)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as fp:  
            fp.write(source_vid.read())
            video_path = str(Path(fp.name))
    
        with open(video_path, "rb") as file:
            video_bytes = file.read()
        st.video(video_bytes)
    else:
        video_path = str(settings.DEFAULT_VIDEO_PATH)
        with open(video_path, "rb") as file:
            video_bytes = file.read()
        st.video(video_bytes)
    ext = '.webm'
    
    
    if st.sidebar.button('Detect Video Objects'):
        progress_bar = st.progress(0, "Detecing Objects...")

        vid_cap = cv2.VideoCapture(video_path)
        # vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        # vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        st_frame = st.empty()
        file_id = uuid.uuid4().hex 

        fps = vid_cap.get(cv2.CAP_PROP_FPS) / counter_max
        frame_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'VP90')
        vid_writer = cv2.VideoWriter(f'{file_id}{ext}',  
                            codec, 
                            fps, (frame_width, frame_height)) 
        
        frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        snakeModel = YOLO(settings.SNAKE_MODEL)
        speciesModel = YOLO(settings.SPECIES_MODEL)
        counter = 0
        log = ""
        expander = st.expander("Detection Results")
        code = expander.code(log, language="json")
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            counter += 1
            if success:
                if counter % counter_max == 0:
                    results = speciesModel.track(image, conf=conf, imgsz=512, persist=True, tracker=tracker, )
                    if len(results[0].boxes) == 0:
                        results = snakeModel.track(image, conf=conf, imgsz=512, persist=True, tracker=tracker)
                    res_plotted = results[0].plot()
                    st_frame.image(res_plotted,
                                caption='Detected Video',
                                channels="BGR",
                                use_column_width=True
                                )
                    vid_writer.write(res_plotted)
                    log += get_string_results(results[0].boxes, results[0].names, frame=vid_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    with expander:
                        code.code(log, language="json")
                    progress_bar.progress(int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES)/frames*100), "Detecing Objects...")
                    counter = 0

            else:
                vid_cap.release()
                vid_writer.release()
                break
        saved_file_path = Path(f'{file_id}{ext}')
        st.sidebar.success(f"Prediction Completed!")
        with open(saved_file_path, "rb") as file:
            video_bytes = file.read()
        st.video(video_bytes)
        progress_bar.progress(100, "Done!")
        os.remove(saved_file_path)
    
        
        
        
            

