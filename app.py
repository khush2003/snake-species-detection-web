# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Snake Species Detection using YOLOv8",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main page heading
st.title("Snake Species Detector")

# Sidebar
st.sidebar.header("Snake Detection Model Config")


confidence = float(st.sidebar.slider(
    "Select Model Confidence", 10, 100, 40)) / 100


species_model_path = Path(settings.SPECIES_MODEL)
snake_model_path = Path(settings.SNAKE_MODEL)


# Load Pre-trained ML Model
try:
    speciesModel = helper.load_model(species_model_path)
    snakeModel = helper.load_model(snake_model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {species_model_path} or {snake_model_path}")
    st.error(ex)


st.sidebar.header("Source Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)


source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    helper.process_image(confidence, speciesModel, snakeModel)
    

#TODO: Video Output + Prediction Log

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, speciesModel)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, snakeModel, speciesModel)


elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, speciesModel, snakeModel)

else:
    st.error("Please select a valid source type!")
