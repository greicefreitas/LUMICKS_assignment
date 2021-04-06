# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2

# path and filename definition:
path = r'C:\Users\20194454\source\repos\LUMICKS'
csvfile_name = 'cell-locations.csv'
video_name = 'fluorescent_microscopy_video_2hz.mp4'
forcemap_name = 'force_map.png'


# variables definition:
frequency = 2  #frequency of the video = 2Hz
cellWindow = 15 #1/2 size of the cell window 


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

# loading video and initial frame:
video = cv2.VideoCapture(path + '//' + video_name)
videoLength = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print( 'Total number of frames:', videoLength)
t_max = videoLength/frequency
print( 'Total time:', t_max)

# reading first frame:
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

# cell_list = [x, y, initial_area, detachment_frame, force]
cell_info = np.zeros((initial_position.shape[0],initial_position.shape[1]+3))
cell_info[:,:-3] = initial_position


for idx,centroid in enumerate(initial_position):
     
    y_i = centroid[1]-cellWindow
    y_f = centroid[1]+cellWindow
    x_i = centroid[0]-cellWindow
    x_f = centroid[0]+cellWindow
    crop_img = initial_frame_gray[y_i:y_f, x_i:x_f]
    retval, bwcroped = cv2.threshold(crop_img, 30, 255, cv2.THRESH_BINARY)
    numberOfPixels_initial = cv2.countNonZero(bwcroped)
    cell_info[idx,2] = numberOfPixels_initial


#--------------------------------------------------------------------------------#
#PROCESSING VIDEO

print("Processing video ...")

while(video.isOpened()): 
    

    ret, frame = video.read()
    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameNumber = video.get(cv2.CAP_PROP_POS_FRAMES)
        print('Frame: ', frameNumber)
        for idx,centroid in enumerate(initial_position):
            if cell_info[idx, 3] == 0 :
                
                # cropp the image around the cell:
                # window definition:
                x_i = centroid[0]-cellWindow
                x_f = centroid[0]+cellWindow
                y_i = centroid[1]-cellWindow
                y_f = centroid[1]+cellWindow

                crop_img = gray[y_i:y_f, x_i:x_f]

                # gray to binary:
                retval, bwcroped = cv2.threshold(crop_img, 30, 255, cv2.THRESH_BINARY)

                # number of cell pixels:
                numberOfPixels = cv2.countNonZero(bwcroped)

                # comparing with original area:
                increase = 100 * float((numberOfPixels - cell_info[idx, 2])/cell_info[idx, 2])

                # if detachment is detected:
                if increase > 20  :
                    cell_info[idx, 3] = frameNumber #store frame number
                    x = int(cell_info[idx, 0])
                    y = int(cell_info[idx, 1])
                    cell_info[idx, 4] = (frameNumber/ frequency)*force_map[x][y]/t_max

    else:
        break

video.release()

#--------------------------------------------------------------------------------#
#RESULTS

# saving information:
with open(path + "//" + 'Results_Information.csv', 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(cell_info)

## plot
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

# reversed cumulative histogram:

nBins = 100
n, bins, patches = ax1.hist(cell_info[:,4], nBins, density=True, histtype='step',
                           cumulative=-1, label='Empirical')

ax1.set_title('Cumulative step histogram')
ax1.set_xlabel('Force (pN)')
ax1.set_ylabel('Likelihood of occurrence (0-1)')
ax1.grid(True)
plt.draw()
fig1.savefig(path + "//" + 'Results_histogram.png')

# boxplot:
ax2.set_title('Force distribution')
ax2.set_xlabel('Dataset')
ax2.set_ylabel('Force (pN)')
ax2.boxplot(cell_info[:,4])
ax2.grid(True)
plt.draw()
fig2.savefig(path + "//" + 'Results_boxplot.png')

plt.show()

