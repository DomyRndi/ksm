from keras_py.utils import get_random_data
from keras_py.face_rec import mask_rec
from keras_py.face_rec import face_rec
from keras_py.mobileNet import MobileNet
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 请加载您最满意的模型 ---------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 dnn.h5 模型，则 model_path = 'results/temp.h5'
model_path = 'results/temp88at.h5'


# ---------------------------------------------------------------------------

def predict(img):
    """
    加载模型和模型预测
    :param img: cv2.imread 图像
    :return: 预测的图片中的总人数、其中佩戴口罩的人数
    """
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    # 将 cv2.imread 图像转化为 PIL.Image 图像，用来兼容测试输入的 cv2 读取的图像（勿删！！！）
    # cv2.imread 读取图像的类型是 numpy.ndarray
    # PIL.Image.open 读取图像的类型是 PIL.JpegImagePlugin.JpegImageFile
    if isinstance(img, np.ndarray):
        # 转化为 PIL.JpegImagePlugin.JpegImageFile 类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    detect = mask_rec(model_path)
    img, all_num, mask_num = detect.recognize(img)

    # -------------------------------------------------------------------------
    return img, all_num, mask_num

img = cv2.imread("./test0.jpg")
img, all_num, mask_num = predict(img)
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(1,1,1)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('test_mask')
ax1.imshow(img)

print("图中的人数有：" + str(all_num) + "个")
print("戴口罩的人数有：" + str(mask_num) + "个")

img = cv2.imread("./test1.jpg")
img, all_num, mask_num = predict(img)
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(1,1,1)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('test_mask')
ax1.imshow(img)

print("图中的人数有：" + str(all_num) + "个")
print("戴口罩的人数有：" + str(mask_num) + "个")

img = cv2.imread("./test2.jpg")
img, all_num, mask_num = predict(img)
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(1,1,1)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('test_mask')
ax1.imshow(img)

print("图中的人数有：" + str(all_num) + "个")
print("戴口罩的人数有：" + str(mask_num) + "个")

img = cv2.imread("./test3.jpg")
img, all_num, mask_num = predict(img)
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(1,1,1)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('test_mask')
ax1.imshow(img)

print("图中的人数有：" + str(all_num) + "个")
print("戴口罩的人数有：" + str(mask_num) + "个")

img = cv2.imread("./test4.jpg")
img, all_num, mask_num = predict(img)
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(1,1,1)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('test_mask')
ax1.imshow(img)

print("图中的人数有：" + str(all_num) + "个")
print("戴口罩的人数有：" + str(mask_num) + "个")
plt.show()