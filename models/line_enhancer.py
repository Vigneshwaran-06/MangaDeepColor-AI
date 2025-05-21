import cv2

def enhance_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
