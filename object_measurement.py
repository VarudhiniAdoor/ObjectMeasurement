import cv2
import numpy as np

def getContours(img, cThr=[100, 100], minArea=5000, filter=4, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDial, kernel, iterations=2)  
    
    contours, _ = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > minArea:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:  # Check if the contour is a rectangle
                finalContours.append([area, approx])
    
    finalContours = sorted(finalContours, key=lambda x: x[0], reverse=True)
    
    if draw:
        for con in finalContours:
            cv2.drawContours(img, [con[1]], -1, (0, 0, 255), 3)
    
    return img, finalContours

def findDistance(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        ret, img = cap.read()
        
        if not ret:
            print("Failed to capture image from webcam.")
            break
        
        imgContours, contours = getContours(img, minArea=5000, filter=4)
        
        if contours:
            for contour in contours:
                rect = contour[1]
                for i in range(4):
                    pt1 = tuple(rect[i][0])
                    pt2 = tuple(rect[(i + 1) % 4][0])
                    cv2.line(imgContours, pt1, pt2, (0, 255, 0), 2)
                
                # Calculate dimensions
                width = findDistance(tuple(rect[0][0]), tuple(rect[1][0]))
                height = findDistance(tuple(rect[1][0]), tuple(rect[2][0]))
                
                # Display dimensions
                x, y, w, h = cv2.boundingRect(rect)
                cv2.putText(imgContours, f'{round(width, 2)} px', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(imgContours, f'{round(height, 2)} px', (x + 10, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Rectangular Object Measurement', imgContours)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
