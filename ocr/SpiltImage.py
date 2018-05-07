# coding=utf-8
from PIL import Image
import os
import time
from gen_printed_char import PreprocessResizeKeepRatioFillBG
import cv2
import numpy as np


#切割图片
def spilt():
    img = Image.open(r'C:\Users\shang\Desktop\image\test1\2.png')
    # region = (5,170,180,200)
    region = (380, 0, 680, 30)
    # 裁切图片
    cropImg = img.crop(region)
    # 保存裁切后的图片
    cropImg.save(r'C:\Users\shang\Desktop\image\result\3.png')

def spilt1(img, a, b, c, d):
    region = (a, b, c, d)
    # 裁切图片
    cropImg = img.crop(region)
    # 保存裁切后的图片
    #cropImg.save(r""+name+"")
    return cropImg
#二值化
def erzhihua(img, rgb):
    #img = Image.open(filename)
    #img = img.convert("RGBA")
    img = img.convert("RGB")
    pixdata = img.load()

    for y in xrange(img.size[1]):
        for x in xrange(img.size[0]):
            if pixdata[x, y][0] > rgb and pixdata[x, y][1] > rgb and pixdata[x, y][2] > rgb:
                pixdata[x, y] = (0, 0, 0)
            else:
                pixdata[x, y] = (255, 255, 255)
    return img

#二值化
def erzhihua1(img, rgb):
    #img = Image.open(filename)
    #img = img.convert("RGBA")
    img = img.convert("RGB")
    pixdata = img.load()

    for y in xrange(img.size[1]):
        for x in xrange(img.size[0]):
            if pixdata[x, y][0] + pixdata[x, y][1] + pixdata[x, y][2] > rgb*3:
                pixdata[x, y] = (255, 255, 255)
            else:
                pixdata[x, y] = (0, 0, 0)
    return img

def  isBlack(img, x, y):
    img = img.convert("RGB")
    pixdata = img.load()
    if pixdata[x, y][0] + pixdata[x, y][1] + pixdata[x, y][2] < 100:
        return True
    else:
        return False;

#找切割点 切割行
def findRowDot(img):
    dotNums = []
    x = img.size[0]/10
    y = img.size[1]
    for yy in xrange(y):
        count = 0;
        for xx in xrange(x):
            if not isBlack(img, xx, yy):
                count = count + 1;
        dotNums.append(count)
    return dotNums


#找切割点 切割行
def findZiFiUpAndFoot(img):
    dotNums = []
    x = img.size[0]
    y = img.size[1]
    for yy in xrange(y):
        count = 0;
        for xx in xrange(x):
            if not isBlack(img, xx, yy):
                count = count + 1;
        dotNums.append(count)
    return dotNums


#找切割点，切割列
def findColDot(img):
    dotNums = []
    x = img.size[0]
    y = img.size[1]
    for xx in xrange(x):
        count = 0;
        for yy in xrange(y):
            if not isBlack(img, xx, yy):
                count = count + 1;
        dotNums.append(count)
    return dotNums


#找切割点，切割列
def checkColDot(img):
    dotNums = []
    x = img.size[0]
    y = img.size[1]
    for xx in xrange(x):
        count = 0;
        for yy in xrange(y):
            if not isBlack(img, xx, yy):
                count = count + 1;
        dotNums.append(count)
    #找出最左不为零和最右不为零的两个点
    dotNums = findCol1(dotNums)
    return dotNums


def ResizeImage(filein, fileout, width, height, type):
    img = Image.open(filein)
    out = img.resize((width, height),Image.ANTIALIAS) #resize image with high-quality
    out.save(fileout, type)



def findRow(dots):
    dotList = []
    d = []
    start = end = 0
    falg = True
    for i in range(len(dots)):
        num = dots[i]
        #当falg是true的时候要找的是不为0的位置
        if falg:
            if num > 0:
                if i>0:
                    start = i-1
                else:
                    start = i
                d.append(start)
                falg = False
                continue
        if not falg:
            if num == 0:
                if end<len(dots):
                    end = i+1
                else:
                    end = i
                falg = True
                d.append(end)
                dotList.append(d)
                d = []
                continue
    return dotList


def findRow1(dots):
    dotList = []
    d = []
    start = end = 0
    falg = True
    for i in range(len(dots)):
        num = dots[i]
        #当falg是true的时候要找的是不为0的位置
        if falg:
            if num > 0:
                start = i
                d.append(start)
                falg = False
                continue
        if not falg:
            if num > 0:
                if end<len(dots):
                    end = i+1
                else:
                    end = i
                continue
    if end == 0:
        end = start + 1
    d.append(end)
    dotList.append(d)
    return dotList


def findCol(dots):
    dotList = []
    d = []
    start = end = 0
    falg = True
    for i in range(len(dots)):
        num = dots[i]
        #当falg是true的时候要找的是不为0的位置
        if falg:
            if num > 0:
                start = i
                d.append(start)
                falg = False
                continue
        if not falg:
            if num == 0 and abs(start-i)>=6:
                end = i
                falg = True
                d.append(end)
                dotList.append(d)
                d = []
                continue
    return dotList


def findCol1(dots):
    dotList = []
    d = []
    start = end = 0
    falg = True
    for i in range(len(dots)):
        num = dots[i]
        # 当falg是true的时候要找的是不为0的位置
        if falg:
            if num > 0:
                start = i
                d.append(start)
                falg = False
                continue
        if not falg:
            if num > 0:
                end = i
                continue
    d.append(end+1)
    dotList.append(d)
    return dotList


#这个方法一般是用来把切割之后长宽不相等的图片调整相等的
def changeImage(img, length):
    x = img.size[0]
    y = img.size[1]
    if length == -1:
        if x>y:
            length = x
        else:
            length = y
    #创建长宽为length的黑色图片，
    newImage = Image.new("RGB", (length+10, length+10), (0, 0, 0))
    #newImage.save("/Users/shangzhen/Desktop/jbxx/jbxx2000.png")
    img = img.convert("RGB")
    pixImg = img.load()
    newImage = newImage.convert("RGB")
    pixNewImage = newImage.load()
    # pixdata[x, y] = (255, 255, 255) 给图片指定像素图上颜色
    if x>y:#原始图片长型
        cha = length - y;
        jiakuan = cha/2
        for i in range(y):
            for j in range(x):
                pixNewImage[j,i+jiakuan] = (pixImg[j,i][0], pixImg[j,i][1], pixImg[j,i][2])
    else:#原始图片高型
        cha = length - x;
        jiakuan = cha/2
        for i in range(y):
            for j in range(x):
                pixNewImage[j+jiakuan, i] = (pixImg[j,i][0], pixImg[j,i][1], pixImg[j,i][2])
    return newImage

#增加边框
def addBorder(img, length):
    x = img.size[0]
    y = img.size[1]
    #创建长宽为length的黑色图片，
    newImage = Image.new("RGB", (x+length, y+length), (0, 0, 0))
    #newImage.save("/Users/shangzhen/Desktop/jbxx/jbxx2000.png")
    img = img.convert("RGB")
    pixImg = img.load()
    newImage = newImage.convert("RGB")
    pixNewImage = newImage.load()
    # pixdata[x, y] = (255, 255, 255) 给图片指定像素图上颜色
    jiakuan = length/2
    for i in range(y):
        for j in range(x):
            pixNewImage[j+jiakuan,i+jiakuan] = (pixImg[j,i][0], pixImg[j,i][1], pixImg[j,i][2])
    return newImage


#缩放图片到指定大小
def suofangImage(img,length,width):
    imga = img.resize((length,width))
    return imga

#把图片缩放成指定大小正方形图片
def ImageChangeSize(image, width, height, marginSize):
    image = addBorder(image, marginSize)
    #image.save("/Users/shangzhen/Desktop/jbxx/jbxx2" + str(j) + "_" + str(z) + ".png")
    image = suofangImage(image, width, height)
    #image.save("/Users/shangzhen/Desktop/jbxx/jbxx2" + str(j) + "_" + str(z) + ".png")
    return image

def spiltImagehanzi(lists):
    newList = []
    for i in range(len(lists)):
        nn = 13
        list = lists[i]
        a = list[0]
        b = list[1]
        if abs(a-b)<6:
            list.remove(i+1)
        elif abs(a-b)>15:
            num = int(round(abs(a-b)/nn))
            if a>b:#要切割开的每个块的宽度
                for j in range(num):
                    l = [b+b*nn,b+(j+1)*nn]
                    newList.append(l)
            else:
                for j in range(num):
                    l = [a+j*nn, a + (j + 1) * nn]
                    #print l
                    newList.append(l)
        else:
            newList.append(list)
    return newList


def analyImage(img):
    list = []
    # 二值化
    img = erzhihua(img, 200)

    dots = findRowDot(img)
    #print dots

    dots = findRow(dots)

    j = 0
    for i in dots:
        #print str(i[0]) + " , " + str(i[1])
        # 横向切割
        img1 = spilt1(img, 0, i[0], img.size[0], i[1])
        j = j + 1

        dots = findColDot(img1)
        dots = findCol(dots)
        # 再多一步过滤,主要过滤没有切开的汉字
        dots = spiltImagehanzi(dots)

        z = 0
        for k in dots:
            kk = k[0]
            image = spilt1(img1, k[0], 0, k[1], img1.size[1])
            # image.save("/Users/shangzhen/Desktop/jbxx/111.png")
            #切割成裸露的字体
            dotss = findZiFiUpAndFoot(image)
            dotss = findRow1(dotss)
            for ii in dotss:
                image = spilt1(image, 0, ii[0], image.size[0], ii[1])
                #检验一下左右是否切的正确，：和数字可能会切偏。
                dotsss = checkColDot(image)
                for d in dotsss:
                    if(d[1]<image.size[0] and d[1]>0):
                        image = spilt1(image, d[0], 0, d[1], image.size[1])
            # image.save("/Users/shangzhen/Desktop/jbxx/111.png")
            #把图片转化成正方形图片
            image = ImageChangeSize(image, 64, 64,2)

            # image.save("/Users/shangzhen/Desktop/jbxx/112.png")
            z = z + 1
            list.append(image)
    return list



#filename = "/Users/shangzhen/Desktop/jbxx/jbxx1.png";
#filename = "/Users/shangzhen/Desktop/dataset/test/03165/13.png";
#img = Image.open(filename)
#img = erzhihua1(img, 10)
#img = suofangImage(img, 20, 20)
#img = suofangImage(img, 64, 64)
#img.save("/Users/shangzhen/Desktop/dataset/test/03165/1313.png")

if __name__ == "__main__":
    # filename = "/Users/shangzhen/Desktop/jbxx/jbxx33.png";
    # img = Image.open(filename)
    # lists = analyImage(img)
    # for i in range(len(lists)):
    #     image = lists[i]
    #     print "/Users/shangzhen/Desktop/jbxx/00" + str(time.time()) + ".png"
    #     image.save("/Users/shangzhen/Desktop/jbxx/" + str(time.time()) + ".png")



    filedir = "/Users/shangzhen/PycharmProjects/CPS-OCR-Engine-master/ocr/tmp/";
    for font_name in os.listdir(filedir):
        path_font_file = os.path.join(filedir, font_name)
        img = Image.open(path_font_file)

        # 二值化
        img = erzhihua(img, 200)

        #把图片转成正方形
        image = ImageChangeSize(img, 64, 64, 2)
        image.save(path_font_file)



#print dots
#qiege(qiegename)
#suofang(qiegename)



