import cv2 as cv

nerfstudio_gt_img = cv.imread('/home/ubuntu/nerfstudio/cache/images/0.jpg')
gaussian_pro_img = cv.imread('/home/ubuntu/GaussianPro/cache/images/0.jpg')

diff_img = nerfstudio_gt_img - gaussian_pro_img

print('Difference sum: ', diff_img.sum())
print('Difference max: ', diff_img.max())
