import cv2
from zgrydnBasicLBP import ZgrydnBasicLBP

image = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

descriptor = ZgrydnBasicLBP(P=8, R=1)
features = descriptor.extract(image)

print(features.shape)
print(features[:10])