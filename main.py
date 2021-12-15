####################################################################################################
# UW-MADISON CS639 FINAL PROJECT - Parking Lot Occupancy w/ Drone Mapping
# Group Members: Jared Krahn (jkrahn2@wisc.edu)
# Fall 2021  
#
# main.py:
#   - Main program application for stitching raw images & counting cars
#
# Other Credits: Pysource (www.pysource.com) - tutorial for how to train yolov3 with Google Collab
####################################################################################################

import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog

####################################################################################################
####################################################################################################
####################################################################################################

running = 1                 # Application running flag
output_stitch  = None       # Stitched image
output_image   = None       # Output image (Stiched + detected cars)
image_scale    = 1.0        # Scale of stitched images
max_image_size = 1920       # Maximum size of stitched image before it is rescaled 
confidence_treshold = 0.1   # Confidence threshold for yolov3 car detection

cwd  = os.getcwd()
root = tk.Tk()
root.withdraw()
input_directory  = filedialog.askdirectory( initialdir = cwd, title = "Select Input Images Folder" ) 
output_directory = os.getcwd() + '\Results'
print( 'Input image directory: ' + input_directory )

####################################################################################################
### Drone image stitching ##########################################################################
####################################################################################################

if running == 1:
    print( 'Beginning Stage: Stitch Raw Images' )

    # Iterate through images in input folder
    raw_images = []
    image_list = os.listdir( input_directory + '/' )

    print( '    Loading raw images!' )
    if len( image_list ) > 1:
        for image in image_list:
            # Load raw images images
            current_image = cv2.imread( input_directory + '/' + image ) 
            raw_images.append( current_image )

        # Create OpenCV2 stitcher & stitch images
        print( '    Stitching images! This may take a while.' )
        stitcher = cv2.Stitcher.create( mode = 1 )
        ( stitcher_status, output_stitch ) = stitcher.stitch( raw_images )

        # Process results
        if( stitcher_status == cv2.STITCHER_OK ):
            print( '    Stitching Successful!' )
        else:
            print( '    STITCHING FAILED! - Ensure that the images are in a good for stitching!' )
            running = 0
    else:
        if len( image_list ) == 1:
            output_stitch = cv2.imread( input_directory + '/' + image_list[ 0 ] )
        else:
            print( '    No image in input directory!' )
            running = 0

    # Resize image if necessary
    if running == 1:
        h, w, rgba = output_stitch.shape

        if w > max_image_size or h > max_image_size: 
            if w > h:
                image_scale = max_image_size / w
            else:
                image_scale = max_image_size / h
                
            output_stitch = cv2.resize( output_stitch, ( 0, 0 ), None, image_scale, image_scale )

####################################################################################################
### Car detection & counting #######################################################################
####################################################################################################

if running == 1:
    print( 'Beginning Stage: Car Detection' )

    # Load yolov3
    net = cv2.dnn.readNet( "yolov3_car.weights", "yolov3.cfg" )

    # Fetch height, width, color
    output_image = output_stitch.copy()
    output_height, output_width, rgba = output_image.shape

    # Detect cars
    print( '    Detecting cars.' )
    layers = net.getLayerNames()
    output_layers = [ layers[i - 1] for i in net.getUnconnectedOutLayers() ]
    blob = cv2.dnn.blobFromImage( output_image, 0.004, ( 416, 416 ), ( 0,0,0 ), True, crop = False )

    net.setInput( blob )
    candidates = net.forward( output_layers )

    # Show car location
    cars        = []
    class_ids   = []
    confidences = []
    for out_candidate in candidates:
        print( '    ' + str( len( out_candidate ) ) + ' candidates detected. Filtering results.' )
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
            x, y, width, height = cars[ i ]
            cv2.circle(  output_image, center = ( x, y ), radius = 6, color = ( 0, 0, 0 ), thickness = -1 )
            cv2.circle(  output_image, center = ( x, y ), radius = 4, color = ( 0, 0, 255 ), thickness = -1 )
            cv2.putText( output_image, str( num_cars ), ( x, y ), cv2.FONT_HERSHEY_PLAIN, 1, ( 0, 0, 0 ), 4 )
            cv2.putText( output_image, str( num_cars ), ( x, y ), cv2.FONT_HERSHEY_PLAIN, 1, ( 255, 255, 255 ), 2 )

    print( str( num_cars ) + " Detected!" )
    cv2.putText( output_image, "Number of cars: " + str( num_cars ), ( 12, 36), cv2.FONT_HERSHEY_COMPLEX, 1, ( 0, 0, 255 ), 2 )
    cv2.imshow( "Number of cars detected: " + str( num_cars ), output_image )
    cv2.imwrite( output_directory + '/output_stitch.jpg', output_stitch )
    cv2.imwrite( output_directory + '/output_cars.jpg', output_image )
    key = cv2.waitKey( 0 )

####################################################################################################
####################################################################################################
####################################################################################################

print( "Application closing" )
cv2.destroyAllWindows()
