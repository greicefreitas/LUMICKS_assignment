# Import libraries
import matplotlib.pyplot as plt
import statistics
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
cellWindowSize = 20 #1/2 size of the cell window 


#--------------------------------------------------------------------------------#
# INITIALIZATION:

print('Initializing ...')

# loading cvs file:
with open(path + '\\' + csvfile_name) as csvfile:
    initialCellPosition = list(csv.reader(csvfile)) 

#Remove header, converting to int and storing in an array
initialCellPosition.pop(0) 
initialCellPosition = [list(map(int, i)) for i in initialCellPosition]
initialCellPosition = np.array(initialCellPosition)

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
for c in initialCellPosition:
    top_left = c[0] - side_length, c[1] - side_length
    bot_right = c[0] + side_length, c[1] + side_length
    img = cv2.rectangle(
        initial_frame, top_left, bot_right, 
        color = (0, 0, 255), thickness = 2
    )
cv2.imshow('Initial position and state of the cells',cv2.resize(img, (1000, 1000))) #Resize to fit the screen (visualization, only)
print( 'Press any key to continue ')
cv2.waitKey(0) 
cv2.destroyAllWindows()

# cell_info = [initial_x, initial_y, initial_area, detachment_frame, force]
cell_info = np.zeros((initialCellPosition.shape[0],initialCellPosition.shape[1]+3))
cell_info[:,:-3] = initialCellPosition

# Processing first frame (initial state):
for idx,c in enumerate(initialCellPosition):
    
    # cell window:
    y_i = c[1]-cellWindowSize
    y_f = c[1]+cellWindowSize
    x_i = c[0]-cellWindowSize
    x_f = c[0]+cellWindowSize
    cellWindow = initial_frame_gray[y_i:y_f, x_i:x_f]
    retval, bwcroped = cv2.threshold(cellWindow, 30, 255, cv2.THRESH_BINARY)

    # number of cell pixels:
    numberOfPixels_initial = cv2.countNonZero(bwcroped)
    cell_info[idx,2] = numberOfPixels_initial

    # update centroid position:
    centroidPosition = cv2.moments(bwcroped)
    cX = int(centroidPosition["m10"] / centroidPosition["m00"])
    cY = int(centroidPosition["m01"] / centroidPosition["m00"])
    initialCellPosition[idx,0] = c[0] + int(cX - cellWindowSize)
    initialCellPosition[idx,1] = c[1] + int(cY - cellWindowSize)



#--------------------------------------------------------------------------------#
#PROCESSING VIDEO

print("Processing video ...")

while(video.isOpened()): 
    

    ret, frame = video.read()
    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameNumber = video.get(cv2.CAP_PROP_POS_FRAMES)
        print('Frame: ', frameNumber)

        for idx,c in enumerate(initialCellPosition):
            if cell_info[idx, 3] == 0:
                
                
                # cropp image (cell window):
             
                x_i = c[0]-cellWindowSize
                x_f = c[0]+cellWindowSize
                y_i = c[1]-cellWindowSize
                y_f = c[1]+cellWindowSize

                cellWindow = gray[y_i:y_f, x_i:x_f]

                # gray to binary:
                retval, bwcroped = cv2.threshold(cellWindow, 30, 255, cv2.THRESH_BINARY)
                
                # updating initial position of the cell:
                centroidPosition = cv2.moments(bwcroped)
                if centroidPosition["m00"] != 0:
                    cX = int(centroidPosition["m10"] / centroidPosition["m00"])
                    cY = int(centroidPosition["m01"] / centroidPosition["m00"])
                    initialCellPosition[idx,0] = c[0] + int(cX - cellWindowSize)
                    initialCellPosition[idx,1] = c[1] + int(cY - cellWindowSize)
                else:
                    cX, cY = 0, 0

                # number of cell pixels:
                numberOfPixels = cv2.countNonZero(bwcroped)

                # comparing with original area:
                increase_area = 100 * float((numberOfPixels - cell_info[idx, 2])/cell_info[idx, 2])

                # if detachment is detected:
                if increase_area > 20:
                    # save frame number
                    cell_info[idx, 3] = frameNumber 
                    # save applied force:
                    cell_info[idx, 4] = (frameNumber/ frequency)*force_map[c[0],c[1]]/t_max 

    else:
        break

video.release()

#--------------------------------------------------------------------------------#
#RESULTS

# saving information:
with open(path + "//" + 'Results_Information.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["X initial", "Y initial", "Area Initial", "Frame Number", "Force"])
    writer.writerows(cell_info)

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

