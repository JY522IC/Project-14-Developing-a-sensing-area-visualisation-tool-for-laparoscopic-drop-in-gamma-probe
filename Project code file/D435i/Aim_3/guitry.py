import cv2 as cv
import numpy as np

# 读入图片
src = cv.imread('test.png')

# 调用cv.putText()添加文字
text = "Nice to see you!"
AddText = src.copy()
cv.putText(AddText, text, (0, 0), cv.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)

# 将原图片和添加文字后的图片拼接起来
res =np.hstack([src, AddText])

# 显示拼接后的图片
cv.imshow('text', res)
cv.waitKey()
cv.destroyAllWindows()