import numpy as np
import csv
import cv2

# path and filename definition:
path = r'C:\Users\20194454\source\repos\LUMICKS'
csvfile_name = 'cell-locations.csv'
video_name = 'fluorescent_microscopy_video_2hz.mp4'
forcemap_name = 'force_map.png'


# variables definition:
frequenncy = 2  #frequency of the video = 2Hz
cell_window = 15 #1/2 size of the cell window 
FrameNumber = 1

#--------------------------------------------------------------------------------#
# INITIALIZATION:

# loading cvs file:
with open(path + '\\' + csvfile_name) as csvfile:
    initial_position = list(csv.reader(csvfile)) 

#Remove header, converting to int and storing in an array
initial_position.pop(0) 
initial_position = [list(map(int, i)) for i in initial_position]
initial_position = np.array(initial_position)

# Loading force map
force_map = cv2.imread(path + '\\' + forcemap_name,0)

# loading the first frame of the video (initial position):
video = cv2.VideoCapture(path + '//' + video_name)
ret, initial_frame = video.read()
initial_frame_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)

# VISUALIZATION: drawing bounding box (BBOX) around the cells

side_length = 8 #Size of the bbox, in pixels (visualization, only)
for centroid in initial_position:
    top_left = centroid[0] - side_length, centroid[1] - side_length
    bot_right = centroid[0] + side_length, centroid[1] + side_length
    img = cv2.rectangle(
        initial_frame, top_left, bot_right, 
        color = (0, 0, 255), thickness = 2
    )
cv2.imshow('Initial position and state of the cells',cv2.resize(img, (1000, 1000))) #Resize to fit the screen (visualization, only)
cv2.waitKey(0) 
cv2.destroyAllWindows()

# cell_list = [x, y, initial_area, detachment_frame]
cell_list = np.zeros((initial_position.shape[0],initial_position.shape[1]+2))
cell_list[:,:-2] = initial_position


for idx,centroid in enumerate(initial_position):
     
    y_i = centroid[1]-cell_window
    y_f = centroid[1]+cell_window
    x_i = centroid[0]-cell_window
    x_f = centroid[0]+cell_window
    crop_img = initial_frame_gray[y_i:y_f, x_i:x_f]
    retval, bwcroped = cv2.threshold(crop_img, 30, 255, cv2.THRESH_BINARY)
    numberOfPixels_initial = cv2.countNonZero(bwcroped)
    cell_list[idx,2] = numberOfPixels_initial


#--------------------------------------------------------------------------------#
#PROCESSING VIDEO

print("Processing video ...")

while(video.isOpened()): 
    

    ret, frame = video.read()
    if ret:
        FrameNumber += 1
        print("Frame",FrameNumber )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for idx,centroid in enumerate(initial_position):
            y_i = centroid[1]-cell_window
            y_f = centroid[1]+cell_window
            x_i = centroid[0]-cell_window
            x_f = centroid[0]+cell_window

            crop_img = gray[y_i:y_f, x_i:x_f]
            retval, bwcroped = cv2.threshold(crop_img, 30, 255, cv2.THRESH_BINARY)
            numberOfPixels = cv2.countNonZero(bwcroped)

            # checking if the area increased:
            increase = 100 * float((numberOfPixels - cell_list[idx, 2])/cell_list[idx, 2])
            if increase > 20 and cell_list[idx, 3] == 0 :
                cell_list[idx, 3] = FrameNumber
    else:
        break

video.release()
######
# Still missing the plot!
# Compute the force!


with open(path + "//" + 'Information.csv', 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(cell_list)
    







