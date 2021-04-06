# LUMICKS Assignment

This program compute the detachement force for the cells. 
Detachment is detected by increasing the area of the cell (more than 20%)

Input:
- csv file containing the initial cells location
- Video
- Force map

Output:
- csv file cointaining:
    - the initial position of the centroid of cell (x,y)
    - the initial area of the cell
    - the detachment moment (video frame number)
    - the force that was being applied to the cell's centroid
- Box Plot of the force distribution
- Adhered cells occurrence histogram 
