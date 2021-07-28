# -*- coding:utf-8 -*-
# @Date :2021/7/28 21:52
# @Author:KittyLess
# @name: ocr_template_match

import cv2
from imutils import contours
import numpy as np
import argparse
import myutils

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,
                help="path to input image")
ap.add_argument("-t", "--template", required=True,
	help="path to template OCR-A image")
args = vars(ap.parse_args())

# 指定信用卡类型
FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}

# 绘图展示
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 读取一个模板图像
img = cv2.imread(args["template"])
#cv_show('img',img)

# 转成灰度图
ref = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv_show('ref',ref)

#二值图像
ref = cv2.threshold(ref,10,255,cv2.THRESH_BINARY_INV)[1]
#cv_show('ref',ref)

# 计算轮廓
#cv2.findContours()函数接受的参数为二值图，
# 即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，
# cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
#ref_：轮廓图像 refCnts：返回一个list中每个元素都是图像中的一个轮廓 hierachy：轮廓之间的关系

refCnts,hierachy = cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,refCnts,-1,(0,0,255),3)
#cv_show('img',img)
print(np.array(refCnts).shape)

# 由于 findContours返回的轮廓的顺序是不确定的 所以要对其进行排序
refCnts = myutils.sort_contours(refCnts,method="left-to-right")[0]
digits = {}

# 遍历每一个轮廓
for (i,c) in enumerate(refCnts):
    # 计算外接矩形并且resize成合适大小
    # x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y+h,x:x+w]
    roi = cv2.resize(roi,(57,88))

    # 每一个数字对应每一个模板
    digits[i] = roi

# 初始化卷积核 返回指定形状和尺寸的结构元素。
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

#读取输入图像 进行预处理
image = cv2.imread(args["image"])
cv_show('image',image)
image = myutils.resize(image,width=300)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#礼帽操作，突出更明亮的区域 开运算图，突出原图像中比周围亮的区域 可以称为滤波器
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat',tophat)

# Sobel 计算梯度
# 当组合为dx=1,dy=0时求x方向的一阶导数，
# 当组合为dx=0,dy=1时求y方向的一阶导数
gradX = cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)

gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

print (np.array(gradX).shape)
cv_show('gradX',gradX)

#通过闭操作（先膨胀，再腐蚀）将数字连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX',gradX)

#THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
# 自适应阈值分割图像
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#cv_show('thresh',thresh)

#再来一个闭操作
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel) #再来一个闭操作
cv_show('thresh',thresh)

# 计算轮廓
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
cv_show('img',cur_img)
locs = []

# 便利轮廓
for (i,c) in enumerate(cnts):
    # 计算矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
    if ar > 2.5 and ar < 4.0:

        if (w > 40 and w < 55) and (h > 10 and h < 20):
            # 符合的留下来
            locs.append((x, y, w, h))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x:x[0])
output = []

# 遍历每个轮廓中的数字
for (i,(gX,gY,gW,gH)) in enumerate(locs):
    groupOutput = []
    # 根据坐标提取每一个组
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    #cv_show('group', group)
    # 预处理
    group = cv2.threshold(group, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group', group)
    # 计算每一组的轮廓
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts,
                                       method="left-to-right")[0]
    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        #cv_show('roi', roi)

        # 计算匹配得分
        scores = []

        for (digit,digitROI) in digits.items():
            # 模板匹配
            # image：待搜索图像
            # templ：模板图像
            # result：匹配结果
            # method：计算匹配程度的方法
            result = cv2.matchTemplate(roi, digitROI,
                                       cv2.TM_CCOEFF)
            # min_val,max_val,min_indx,max_indx
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))

    # 画出来
    cv2.rectangle(image, (gX - 5, gY - 5),
                  (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 得到结果
    output.extend(groupOutput)

# 打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)