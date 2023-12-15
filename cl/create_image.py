import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def cv2_imshow(a):
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 1920 x 200 pixel
width = 1920
height = 200
a = np.ones((height, width, 3))

snake_red = np.array([7, 26, 214]) / 256
snake_green = np.array([103, 169, 3]) / 256

rgb = 123, 138, 139
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        a[i][j] = np.array([*rgb]) / 255 # RGB 3 169 103 -> GBR BGR?



img = Image.fromarray((a * 255).astype(np.uint8))
# plt.imshow(img)
# plt.show()
img.save('flatly_secondary.jpg')

# rgb_a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
# cv2.imwrite('flatly_secondary.png', rgb_a)
