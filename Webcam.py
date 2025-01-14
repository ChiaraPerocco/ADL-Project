# How to capture image from webcam using opencv
# Source: https://note.nkmk.me/en/python-opencv-camera-to-still-image/
#         https://www.youtube.com/watch?time_continue=682&v=5ZrtPi-7EN0&embeds_referring_euri=https%3A%2F%2Fwww.bing.com%2F&embeds_referring_origin=https%3A%2F%2Fwww.bing.com&source_ve_path=Mjg2NjY

if True:
    import cv2
    import os

    current_dir = os.path.dirname(__file__)
    print(current_dir)

    # relative path of taken images
    # Set the directory path where the captured images will be saved
    dir_path = os.path.join(current_dir, "webcam_images")

    # Ensure the directory exists
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Define a function that captures and saves an image from the webcam when a specific key is pressed
    def save_frame_camera_key(basename, dir_path, ext='jpg', delay=1):

        # Initialize the connection to the webcam (camera index 0 for the default webcam)
        video_capture = cv2.VideoCapture(0)

        # Check if the camera was successfully opened
        if not video_capture.isOpened():
            print("Error: Could not open the camera.")
            return


        # Create the base path for the images to be saved
        base_path = os.path.join(dir_path, basename)

        # Check if the directory contains any files and delete them if necessary
        if os.path.exists(base_path + f'_0.{ext}'):
            os.remove(base_path + f'_0.{ext}')

        n = 0
        while True:
            # Read the current frame from the webcam
            ret, frame = video_capture.read()
            # Display the current frame in a window named "Webcam"
            cv2.imshow("Webcam", frame)
            # Wait for a key press; check every "delay" milliseconds if a key is pressed
            key = cv2.waitKey(delay) & 0xFF
            # If the 'y' key is pressed, save the frame and exit the loop
            if key == ord('y'):
                cv2.imwrite('{}_{}.{}'.format(base_path, n, ext), frame)
                n += 1
                break
            # If the 'n' key is pressed, exit the loop without saving
            elif key == ord('n'):
                break
            
        # Close the opened webcam window
        cv2.destroyWindow("Webcam")


    #save_frame_camera_key('camera_capture', dir_path=dir_path)

