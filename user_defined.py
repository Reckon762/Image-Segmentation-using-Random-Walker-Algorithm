import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt


seg_col = [[255, 255, 255], [0, 0, 0]]
sigma = 0.35

def down(x):
    return int(x*sigma)

def up(x):
    return int(x/sigma)

def mouse_callback(event, x, y, flags, params):

    if event == 1:
        clicks.append([x, y])
        print(clicks)


def getVal(y, x, ar):
    if x < 0 or y < 0 or y >= ar.shape[0] or x >= ar.shape[1]:
        return np.array([-1000.0, -1000.0, -1000.0])
    else:
        return ar[y, x, :]

img = cv.imread("Dataset\\9.png")

labelledPixelsXY = []
noPixels = 30

# Interactive input for initially marked pixels
for n in range(2):
    print("Select 10 pixels in image for segment -", n)
    cv.imshow("Input Image", img)
    clicks = []
    cv.setMouseCallback('Input Image', mouse_callback)
    while True:
        if len(clicks) == noPixels:
            break
        cv.waitKey(1)
    labelledPixelsXY.append(clicks)
    clicks = []

print(labelledPixelsXY)

# Resize the image to reduce processing time
imgOriginal = np.array(img)
img = img/255.0
img = cv.resize(img, (int(img.shape[1]*sigma)+1,
                      int(img.shape[0]*sigma)+1))

initiallyMarked = np.zeros((img.shape[0], img.shape[1]), dtype=np.int)
initiallyMarked.fill(-1)
segments = np.zeros((img.shape[0], img.shape[1]), dtype=np.int)
segments.fill(-1)
cum_prob = np.zeros((img.shape[0],
                     img.shape[1], 4), dtype=np.float)

# Generating the transition probabilites based on pixel similarity
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        urdl = [getVal(y-1, x, img), getVal(y, x+1, img), getVal(y+1, x, img), getVal(y, x-1, img)]
        nonNormalizedProbURDL = []
        for a in range(4):
            tt = np.mean(np.abs(urdl[a]-img[y, x, :]))
            tt = np.exp(-1*np.power(tt, 2))
            nonNormalizedProbURDL.append(tt)
        nonNormalizedProbURDL = np.array(nonNormalizedProbURDL)
        normalizedProbURDL = \
            nonNormalizedProbURDL / np.sum(nonNormalizedProbURDL)
        cum_prob[y, x, 0] = normalizedProbURDL[0]

        for a in range(1, 4):
            cum_prob[y, x, a] =\
                cum_prob[y, x, a-1] +\
                normalizedProbURDL[a]

for s in range(2):
    for a in range(len(labelledPixelsXY[s])):
        print(initiallyMarked.shape, down(labelledPixelsXY[s][a][1]), down(labelledPixelsXY[s][a][0]))
        initiallyMarked[down(labelledPixelsXY[s][a][1]), down(labelledPixelsXY[s][a][0])] = s
        segments[down(labelledPixelsXY[s][a][1]), down(labelledPixelsXY[s][a][0])] = s

# Applying Random Walker Algorithm
for y in range(segments.shape[0]):
    for x in range(segments.shape[1]):
        if segments[y][x] == -1:
            yy = y
            xx = x
            while (initiallyMarked[yy, xx] == -1):
                rv = random.random()
                if cum_prob[yy, xx, 0] > rv:
                    yy -= 1
                elif cum_prob[yy, xx, 1] > rv:
                    xx += 1
                elif cum_prob[yy, xx, 2] > rv:
                    yy += 1
                else:
                    xx -= 1
            segments[y, x] = initiallyMarked[yy, xx]
        print("walked", y, x)

outputImg = np.array(imgOriginal)
for y in range(outputImg.shape[0]):
    for x in range(outputImg.shape[1]):
        outputImg[y, x] = seg_col[segments[down(y), down(x)]]

plt.imshow(outputImg, cmap='gray')
plt.show()
