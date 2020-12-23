import os
import glob
import numpy as np
import cv2
from PIL import Image, ImageOps
import Image_Augmentation as IA
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


input_dir = 'trainImg' # 学習の原画像が入っているフォルダ名
output_dir = "output" # 拡張処理後の出力フォルダ名
num = 5 # 拡張する画像枚数

generator = ImageDataGenerator(
                rotation_range=90, # 回転する角度を90°までに設定
                width_shift_range=0.1, # 水平方向にランダムでシフト
                height_shift_range=0.1, # 垂直方向にランダムでシフト
                zoom_range=0.3, # 拡大・縮小する範囲
                channel_shift_range=50.0, # ランダム値を画素値に加える
                horizontal_flip=False, # 垂直方向にランダムで反転
                vertical_flip=True # 水平方向にランダムで反転
                )

def canny(image):
    # 拡張処理後の出力フォルダ作成
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)

    # 指定したフォルダ名の中にある画像リストを得る
    images = glob.glob(os.path.join(input_dir + "/*")) 

    # リストの画像順に拡張処理を適用する
    for i in range(len(images)):

        # 画像のファイル名の取得
        path, text = os.path.splitext(os.path.basename(images[i]))

        # 画像の読み込み
        img = load_img(images[i])        

        # 拡張処理に適用すための画像の変形処理
        img = img_to_array(img)
        ar_img = np.expand_dims(img, axis=0)

        # 画像の拡張
        test = IA.process_imgs(generator, ar_img, output_dir, path, num)


    image_1 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) #時計回りに９０°回転
    image_2 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) #反時計回りに９０°回転
    image_3 = cv2.rotate(image, cv2.ROTATE_180) #時計回りに１８０°回転
    return image_1, image_2, image_3