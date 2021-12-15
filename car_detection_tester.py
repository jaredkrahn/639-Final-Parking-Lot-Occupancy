####################################################################################################
# UW-MADISON CS639 FINAL PROJECT - Parking Lot Occupancy w/ Drone Mapping
# Group Members: Jared Krahn (jkrahn2@wisc.edu)
# Fall 2021  
#
# car_detection_tester.py:
#   - Application to test car detection accuracy over multiple separate images (within a folder)
#
# Other Credits: Pysource (www.pysource.com) - tutorial for how to train yolov3 with Google Collab
####################################################################################################

import cv2
import os
import numpy as np
import tkinter as tk
import glob
from tkinter import filedialog

####################################################################################################
####################################################################################################
####################################################################################################

confidence_treshold = 0.1   # Confidence threshold for yolov3 car detection
image_scale    = 1.0        # Scale of stitched images
max_image_size = 1920       # Maximum size of stitched image before it is rescaled 
draw_boxes     = 1

cwd  = os.getcwd()
root = tk.Tk()
root.withdraw()
input_directory  = filedialog.askdirectory( initialdir = cwd, title = "Select Input Images Folder" ) 

####################################################################################################
### Car detection & counting test ##################################################################
####################################################################################################

# Load yolov3
net = cv2.dnn.readNet( "yolov3_car.weights", "yolov3.cfg" )
layers = net.getLayerNames()
output_layers = [ layers[i - 1] for i in net.getUnconnectedOutLayers() ]

for image in glob.glob(input_directory + "\*.jpg"):

    # Fetch height, width, color
    output_image = cv2.imread( image )
    output_height, output_width, rgba = output_image.shape

    # Resize image if too big
    if output_width > max_image_size or output_height > max_image_size: 
        if output_width > output_height:
            image_scale = max_image_size / output_width
        else:
            image_scale = max_image_size / output_height
            
        output_image = cv2.resize( output_image, ( 0, 0 ), None, image_scale, image_scale )
        output_height, output_width, rgba = output_image.shape # Fetch new size

    # Detect cars
    blob = cv2.dnn.blobFromImage( output_image, 0.004, ( 416, 416 ), ( 0,0,0 ), True, crop = False )
    net.setInput( blob )
    candidates = net.forward( output_layers )

    # Show car locations
    cars        = []
    class_ids   = []
    confidences = []
    for out_candidate in candidates:
        for candidate in out_candidate:
            scores = candidate[ 5: ]
            this_class = np.argmax( scores ) 

            # Filter candidates by confidence threshold
            confidence = scores[ this_class ]
            if confidence > confidence_treshold:
                confidences.append( float( confidence ) )
                class_ids.append( this_class )

                # Locate car position in image
                x      = int( candidate[0] * output_width )
                y      = int( candidate[1] * output_height )
                width  = int( candidate[2] * output_width )
                height = int( candidate[3] * output_height )
                cars.append( [ x, y, width, height ]  )

    # Count detected cars & display their locations
    num_cars = 0   
    for i in range( len( cars ) ):
        if i in cv2.dnn.NMSBoxes( cars, confidences, 0.05, 0.025 ):
            num_cars += 1 # Increment car counter

            # Draw circle & car number over detected cars
            if draw_boxes == 1:
                x, y, width, height = cars[ i ]
                rX1 = round(x - width/2)
                rX2 = round(x + width/2)
                rY1 = round(y - height/2)
                rY2 = round(y + height/2)

            cv2.rectangle( output_image, ( rX1, rY1 ), ( rX2, rY2 ), ( 0, 0, 255 ), 1)
            cv2.circle(  output_image, center = ( x, y ), radius = 6, color = ( 0, 0, 0 ), thickness = -1 )
            cv2.circle(  output_image, center = ( x, y ), radius = 4, color = ( 0, 0, 255 ), thickness = -1 )
            cv2.putText( output_image, str( num_cars ), ( x, y ), cv2.FONT_HERSHEY_PLAIN, 1, ( 0, 0, 0 ), 4 )
            cv2.putText( output_image, str( num_cars ), ( x, y ), cv2.FONT_HERSHEY_PLAIN, 1, ( 255, 255, 255 ), 2 )

    cv2.putText( output_image, "Number of cars: " + str( num_cars ), ( 12, 36), cv2.FONT_HERSHEY_COMPLEX, 1, ( 0, 0, 255 ), 2 )
    cv2.imshow( "Car Detection Test", output_image )
    key = cv2.waitKey( 0 )

####################################################################################################
####################################################################################################
####################################################################################################

cv2.destroyAllWindows()
