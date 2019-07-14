import cv2
import numpy as np
from PIL import Image, ImageTk, ImageChops

pil_img = Image.open('activations_grid.png').convert('RGB')
cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

cv2.namedWindow('viz', cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty('viz', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('viz', cv_img)
cv2.waitKey()
cv2.destroyAllWindows()
