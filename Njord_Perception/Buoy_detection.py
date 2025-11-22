import cv2
import pyzed.sl as sl
import numpy as np
import sys
import math

def main():
    # Define filter values using BGR values
    red_Filter_Lower = np.array([0, 0, 75])
    red_Filter_Upper = np.array([75, 50, 255])
    
    green_Filter_Lower = np.array([0, 50, 0])
    green_Filter_Upper = np.array([75, 255, 25])

    # Create a Camera object
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720

    # Open the camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open camera, exiting.")
        exit()

    runtime_parameters = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    image_depth = sl.Mat()

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.retrieve_image(image_depth, sl.VIEW.DEPTH)
            
            cv_img = image.get_data()
            red_Matrix = cv_img.copy()
            green_Matrix = cv_img.copy()
            red_Matrix = red_Matrix[:,:,[0,1,2]]
            green_Matrix = green_Matrix[:,:,[0,1,2]]

            # Create masks for red and green
            red_Mask = cv2.inRange(red_Matrix, red_Filter_Lower, red_Filter_Upper)
            green_Mask = cv2.inRange(green_Matrix, green_Filter_Lower, green_Filter_Upper)

            # Reduce noise and filter small objects
            red_Mask = process_mask(red_Mask)
            green_Mask = process_mask(green_Mask)

            # Detect and display objects
            detect_and_display_objects(red_Mask, depth, "Red Object", cv_img, (0,0,255))
            #detect_and_display_objects(red_Mask, image_depth, "Red Object", cv_img, (0,0,255)) #Gir vektor istede for avstand i [mm]
            detect_and_display_objects(green_Mask, depth, "Green Object", cv_img, (0,255,0))
            
            cv2.imshow("Camera View", cv_img)

            if cv2.waitKey(30) >= 0:
                break

def process_mask(mask, min_area=1000):
    mask = reduce_noise(mask)
    mask = filter_small_objects(mask, min_area)
    return mask

def reduce_noise(mask):
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opening

def filter_small_objects(mask, min_area=1000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > min_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, cv2.FILLED)
    return filtered_mask

def calculate_angle(image_width, object_x):
    # Camera parameters (example values)
    fov_horizontal_degrees = 110  # Horizontal field of view in degrees

    # Calculate angle relative to camera center
    center_x = image_width / 2
    pixel_offset = object_x - center_x
    angle_per_pixel = fov_horizontal_degrees / image_width
    angle_offset_degrees = pixel_offset * angle_per_pixel

    return angle_offset_degrees

def draw_contours(image, contours, color):
    cv2.drawContours(image, contours, -1, color, 2)
                     
def detect_and_display_objects(mask, depth, object_name, cv_image, color):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    draw_contours(cv_image, contours, color)
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            depth_value = depth.get_value(cx, cy)[1]

            # Calculate angle relative to camera center
            angle_degrees = calculate_angle(cv_image.shape[1], cx)
            
            # Print angle and depth information
            print(f"{object_name} at ({cx}, {cy}) with depth {(depth_value/1000):.2f} m and angle {(angle_degrees):.2f} degrees")

            # Example: Calculate desired coordinates for navigation
            if depth_value is not None:
                desired_coordinates = calculate_desired_coordinates(depth_value, angle_degrees)
                print("Desired Coordinates:", desired_coordinates)



def calculate_desired_coordinates(depth_value, angle_degrees):
    # Example: Calculate desired coordinates for navigation based on object depth and angle
    # This function should calculate the desired navigation coordinates relative to the camera or a reference frame
    # You may perform coordinate transformations or use other methods depending on your setup
    desired_x = depth_value * math.cos(math.radians(angle_degrees))
    desired_y = depth_value * math.sin(math.radians(angle_degrees))
    desired_coordinates = (desired_x, desired_y, depth_value)
    return desired_coordinates


if __name__ == "__main__":
    main()
