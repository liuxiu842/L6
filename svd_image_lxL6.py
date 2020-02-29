import numpy as np
from scipy.linalg import svd
from PIL import Image
import matplotlib.pyplot as plt


# 取前k个特征，对图像进行还原
def get_image_feature(s, k):
    # 对于S，只保留前K个特征值
    s_temp = np.zeros(s.shape[0])
    print("s_temp1 is :", s_temp)
    s_temp[0:k] = s[0:k]
    print("s_temp2 is :", s_temp)
    s = s_temp * np.identity(s.shape[0])
    print("s is:",s)
    # 用新的s_temp，以及p,q重构A
    temp = np.dot(p,s)
    temp = np.dot(temp,q)
    plt.imshow(temp, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()
    print("temp is", temp)
    print("A-temp is", A-temp)
    print("A is", A)


# 加载256色图片
image = Image.open('./new.png')
A = np.array(image)
print("A is :", A.shape)
# 显示原图像
plt.imshow(A, cmap=plt.cm.gray, interpolation='nearest')
plt.show()
# 对图像矩阵A进行奇异值分解，得到p,s,q
p,s,q = svd(A, full_matrices=False)
print("s is :", s)
print("p is :", p)
print("q is :", q)
# 取前k个特征，对图像进行还原
#get_image_feature(s, 5)
get_image_feature(s, 100)
get_image_feature(s, 500)

