import tempfile
import numpy as np

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from ultralytics import YOLO
from detection import create_colors_info, detect



def main():

    st.set_page_config(page_title="AI Powered Web Application for Football Tactical Analysis", layout="wide", initial_sidebar_state="expanded")
    st.title("Football Players Detection With Team Prediction & Tactical Map")
    st.subheader(":red[Works only with Tactical Camera footage]")

    st.sidebar.title("Main Settings")
    ## Sidebar Setup
    st.sidebar.subheader("Video Upload")
    input_vide_file = st.sidebar.file_uploader('Upload a video file', type=['mp4','mov', 'avi', 'm4v', 'asf'])

    if not input_vide_file:
        st.warning("⚠️ Please upload a video to continue.")
        return
    else:
        tempf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tempf.write(input_vide_file.read())
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()

        st.sidebar.text('Uploaded video')
        st.sidebar.video(demo_bytes)
    
    # Load the YOLOv8 players detection model
    model_players = YOLO("../models/Yolo8L Players/weights/best.pt", task="detect")
    # Load the YOLOv8 field keypoints detection model
    model_keypoints = YOLO("../models/Yolo8M Field Keypoints/weights/best.pt", task="pose")


    st.sidebar.markdown('---')
    st.sidebar.subheader("Team Names")
    team1_name = st.sidebar.text_input(label='First Team Name', value="")
    team2_name = st.sidebar.text_input(label='Second Team Name', value="")
    st.sidebar.markdown('---')

    ## Page Setup
    tab1, tab2, tab3 = st.tabs(["How to use?", "Team Colors", "Model Hyperparameters & Detection"])
    with tab1:
        st.header(':blue[Welcome!]')
        st.subheader('Main Application Functionalities:', divider='blue')
        st.markdown("""
                    1. Football players, referee, and ball detection.
                    2. Players team prediction.
                    3. Estimation of players and ball positions on a tactical map.
                    4. Ball Tracking.
                    """)
        st.subheader('How to use?', divider='blue')
        st.markdown("""
                    **There are two demo videos that are automaticaly loaded when you start the app, alongside the recommended settings and hyperparameters**
                    1. Upload a video to analyse, using the sidebar menu "Browse files" button.
                    2. Enter the team names that corresponds to the uploaded video in the text fields in the sidebar menu.
                    3. Access the "Team colors" tab in the main page.
                    4. Select a frame where players and goal keepers from both teams can be detected.
                    5. Follow the instruction on the page to pick each team colors.
                    6. Go to the "Model Hyperpramerters & Detection" tab, adjust hyperparameters and select the annotation options. (Default hyperparameters are recommended)
                    7. Run Detection!
                    8. If "save outputs" option was selected the saved video can be found in the "outputs" directory
                    """)
        st.write("Version 0.0.1")

    with tab2:
        # Stable internal keys (do NOT use team names)
        KEY_T1_P = "team1_p_color"
        KEY_T1_GK = "team1_gk_color"
        KEY_T2_P = "team2_p_color"
        KEY_T2_GK = "team2_gk_color"

        t1col1, t1col2 = st.columns([1,1])
        with t1col1:
            cap_temp = cv2.VideoCapture(tempf.name)
            frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_nbr = st.slider(label="Select frame", min_value=1, max_value=frame_count, step=1, help="Select frame to pick team colors from")
            cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr)
            success, frame = cap_temp.read()
            with st.spinner('Detecting players in selected frame..'):
                results = model_players(frame, conf=0.7)
                bboxes = results[0].boxes.xyxy.cpu().numpy()
                labels = results[0].boxes.cls.cpu().numpy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections_imgs_list = []
                detections_imgs_grid = []
                padding_img = np.ones((80,60,3),dtype=np.uint8)*255
                for i, j in enumerate(list(labels)):
                    if int(j) == 0:
                        bbox = bboxes[i,:]
                        obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                        obj_img = cv2.resize(obj_img, (60,80))
                        detections_imgs_list.append(obj_img)
                # handle empty or odd counts safely
                if len(detections_imgs_list) == 0:
                    concat_det_imgs = padding_img
                else:
                    half = len(detections_imgs_list)//2
                    detections_imgs_grid.append([detections_imgs_list[i] for i in range(half)])
                    detections_imgs_grid.append([detections_imgs_list[i] for i in range(half, len(detections_imgs_list))])
                    if len(detections_imgs_list)%2 != 0:
                        detections_imgs_grid[0].append(padding_img)
                    concat_det_imgs_row1 = cv2.hconcat(detections_imgs_grid[0])
                    concat_det_imgs_row2 = cv2.hconcat(detections_imgs_grid[1])
                    concat_det_imgs = cv2.vconcat([concat_det_imgs_row1,concat_det_imgs_row2])

            st.write("Detected players")
            # Radio options are human-friendly labels; we map them to stable keys
            radio_display_to_key = {
                f"{team1_name or 'Team 1'} Player": KEY_T1_P,
                f"{team1_name or 'Team 1'} GK": KEY_T1_GK,
                f"{team2_name or 'Team 2'} Player": KEY_T2_P,
                f"{team2_name or 'Team 2'} GK": KEY_T2_GK,
            }
            radio_options = list(radio_display_to_key.keys())

            value = streamlit_image_coordinates(concat_det_imgs, key="numpy")
            st.markdown('---')
            active_color_display = st.radio(label="Select which team color to pick from the image above",
                                            options=radio_options, horizontal=True,
                                            help="Choose team color you want to pick and click on the image above to pick the color. Colors will be displayed in boxes below.")

            # When clicking the image, store the hex color under the mapped stable key
            if value is not None:
                picked_color = concat_det_imgs[value['y'], value['x'], :]
                mapped_key = radio_display_to_key[active_color_display]
                st.session_state[mapped_key] = '#%02x%02x%02x' % tuple(picked_color)

            st.write("Boxes below can be used to manually adjust selected colors.")
            cp1, cp2, cp3, cp4 = st.columns([1,1,1,1])

            # Use stable keys when getting/setting values for color pickers
            with cp1:
                hex_color_1 = st.session_state.get(KEY_T1_P, "#1E2530")  # default dark blue
                team1_p_color = st.color_picker(label=' ', value=hex_color_1, key='t1p')
                st.session_state[KEY_T1_P] = team1_p_color
            with cp2:
                hex_color_2 = st.session_state.get(KEY_T1_GK, "#FFD700")  # default yellow
                team1_gk_color = st.color_picker(label=' ', value=hex_color_2, key='t1gk')
                st.session_state[KEY_T1_GK] = team1_gk_color
            with cp3:
                hex_color_3 = st.session_state.get(KEY_T2_P, "#FFFFFF")  # default white
                team2_p_color = st.color_picker(label=' ', value=hex_color_3, key='t2p')
                st.session_state[KEY_T2_P] = team2_p_color
            with cp4:
                hex_color_4 = st.session_state.get(KEY_T2_GK, "#00FF00")  # default green
                team2_gk_color = st.color_picker(label=' ', value=hex_color_4, key='t2gk')
                st.session_state[KEY_T2_GK] = team2_gk_color

            st.markdown('---')

        with t1col2:
            extracted_frame = st.empty()
            extracted_frame.image(frame, use_column_width=True, channels="BGR")

    # When creating colors_dic, use the stable keys
    colors_dic, color_list_lab = create_colors_info(
        team1_name,
        st.session_state.get("team1_p_color", "#1E2530"),
        st.session_state.get("team1_gk_color", "#FFD700"),
        team2_name,
        st.session_state.get("team2_p_color", "#FFFFFF"),
        st.session_state.get("team2_gk_color", "#00FF00"),
    )


    with tab3:
        t2col1, t2col2 = st.columns([1,1])
        with t2col1:
            player_model_conf_thresh = st.slider('PLayers Detection Confidence Threshold', min_value=0.0, max_value=1.0, value=0.6)
            keypoints_model_conf_thresh = st.slider('Field Keypoints PLayers Detection Confidence Threshold', min_value=0.0, max_value=1.0, value=0.7)
            keypoints_displacement_mean_tol = st.slider('Keypoints Displacement RMSE Tolerance (pixels)', min_value=-1, max_value=100, value=7,
                                                         help="Indicates the maximum allowed average distance between the position of the field keypoints\
                                                           in current and previous detections. It is used to determine wether to update homography matrix or not. ")
            detection_hyper_params = {
                0: player_model_conf_thresh,
                1: keypoints_model_conf_thresh,
                2: keypoints_displacement_mean_tol
            }
        with t2col2:
            num_pal_colors = st.slider(label="Number of palette colors", min_value=1, max_value=5, step=1, value=3,
                                    help="How many colors to extract form detected players bounding-boxes? It is used for team prediction.")
            st.markdown("---")
            save_output = st.checkbox(label='Save output', value=False)
            if save_output:
                output_file_name = st.text_input(label='File Name (Optional)', placeholder='Enter output video file name.')
            else:
                output_file_name = None
        st.markdown("---")

        
        bcol1, bcol2 = st.columns([1,1])
        with bcol1:
            nbr_frames_no_ball_thresh = st.number_input("Ball track reset threshold (frames)", min_value=1, max_value=10000,
                                                     value=30, help="After how many frames with no ball detection, should the track be reset?")
            ball_track_dist_thresh = st.number_input("Ball track distance threshold (pixels)", min_value=1, max_value=1280,
                                                        value=100, help="Maximum allowed distance between two consecutive balls detection to keep the current track.")
            max_track_length = st.number_input("Maximum ball track length (Nbr. detections)", min_value=1, max_value=1000,
                                                        value=35, help="Maximum total number of ball detections to keep in tracking history")
            ball_track_hyperparams = {
                0: nbr_frames_no_ball_thresh,
                1: ball_track_dist_thresh,
                2: max_track_length
            }
        with bcol2:
            st.write("Annotation options:")
            bcol21t, bcol22t = st.columns([1,1])
            with bcol21t:
                show_k = st.toggle(label="Show Keypoints Detections", value=False)
                show_p = st.toggle(label="Show Players Detections", value=True)
            with bcol22t:
                show_pal = st.toggle(label="Show Color Palettes", value=True)
                show_b = st.toggle(label="Show Ball Tracks", value=True)
            plot_hyperparams = {
                0: show_k,
                1: show_pal,
                2: show_b,
                3: show_p
            }
            st.markdown('---')
            bcol21, bcol22, bcol23, bcol24 = st.columns([1.5,1,1,1])
            with bcol21:
                st.write('')
            with bcol22:
                ready = True if (team1_name == '') or (team2_name == '') else False
                start_detection = st.button(label='Start Detection', disabled=ready)
            with bcol23:
                stop_btn_state = True if not start_detection else False
                stop_detection = st.button(label='Stop Detection', disabled=stop_btn_state)
            with bcol24:
                st.write('')


    stframe = st.empty()
    cap = cv2.VideoCapture(tempf.name)
    status = False

    if start_detection and not stop_detection:
        st.toast(f'Detection Started!')
        status = detect(cap, stframe, output_file_name, save_output, model_players, model_keypoints,
                         detection_hyper_params, ball_track_hyperparams, plot_hyperparams,
                           num_pal_colors, colors_dic, color_list_lab)
    else:
        try:
            # Release the video capture object and close the display window
            cap.release()
        except:
            pass
    if status:
        st.toast(f'Detection Completed!')
        cap.release()


if __name__=='__main__':
    try:
        main()
    except SystemExit:
        pass