#読み込んだ画像を表示するプログラム
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('test.png') #写真の名前(同じディレクトリに)
print(type(img))
# <class 'numpy.ndarray'>

print(img.shape)
# (225, 400, 3)

img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite('data/dst/lena_cv_rotate_90_clockwise.jpg', img_rotate_90_clockwise)
plt.imshow(img_rotate_90_clockwise)
plt.show() #画面に表示
# True
print(img_rotate_90_clockwise.shape)
print(img_rotate_90_clockwise)

img_rotate_90_counterclockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imwrite('data/dst/lena_cv_rotate_90_counterclockwise.jpg', img_rotate_90_counterclockwise)
plt.imshow(img_rotate_90_counterclockwise)
plt.show()
# True

img_rotate_180 = cv2.rotate(img, cv2.ROTATE_180)
cv2.imwrite('data/dst/lena_cv_rotate_180.jpg', img_rotate_180)
plt.imshow(img_rotate_180)
plt.show()
# True
