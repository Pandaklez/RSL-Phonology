# This code compares four comparison metrics of histograms for cv.compareHist function

import cv2

def hist(img):
    """
    Returns a histogram analysis of a single image (in this case, a video frame)
    """
    return cv2.calcHist([img],[0],None,[256],[0,256])

src_base = cv2.imread('love-q.jpg')
src_test1 = cv2.imread('love-same.jpg')
src_test2 = cv2.imread('love-move.jpg')

# src_base = cv2.imread('sun-blink.jpg')
# src_test1 = cv2.imread('sun-no-blink.jpg')
# src_test2 = cv2.imread('sun-hold.jpg')

hist_base = hist(src_base)
cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
hist_test1 = hist(src_test1)
cv2.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
hist_test2 = hist(src_test2)
cv2.normalize(hist_test2, hist_test2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

for compare_method in range(4):
    base_base = cv2.compareHist(hist_base, hist_base, compare_method)
    base_test1 = cv2.compareHist(hist_base, hist_test1, compare_method)
    base_test2 = cv2.compareHist(hist_base, hist_test2, compare_method)
    print('Method:', compare_method, 'Perfect, Base-Test(1), Base-Test(2) :',\
          base_base, '/', base_test1, '/', base_test2)

"""
fall-in-love

Base-Test(1) = almost the same image as the query one
Base-Test(2) = movement after the hold, so different from the query image
                        
                        Perfect, Base-Test(1), Base-Test(2) :
Method: 0 Correlation   1.0 / 0.9841678666200665 / 0.9785676075887974     # very small diff
Method: 1 Chi-squared   0.0 / 2958.5602355009546 / 4112.501558459098      # chi-squared is no good in general
Method: 2 Intersection  76800.0 / 71844.0 / 71166.0                       # very small diff
Method: 3 Bhattacharyya 0.0 / 0.06601046575809169 / 0.09677062321901717   # very small diff 
"""

"""
sun

Base-Test(1) = image with opened eyes; should be the same as image with closed eyes, as query image
Base-Test(2) = hold, should be different from query image

                        Perfect, Base-Test(1), Base-Test(2) :
Method: 0 Correlation   1.0 / 0.9907733744652303 / 0.9200271778196187     #
Method: 1 Chi-squared   0.0 / 1343.9209206318706 / 11841.32658791789      # wrong result
Method: 2 Intersection  76800.0 / 73408.0 / 64931.0                       #
Method: 3 Bhattacharyya 0.0 / 0.04741276067617973 / 0.14177961801806496   # is not sensitive to eyeblinks! nice!
"""

# Correlation is not sensitive to eyeblinks at all, but it gives very small difference between movement and hold
# Bhattacharyya is worth a try. It looks almost like the reverse correlation metric, however, the differences a one tiny bit bigger

"""
sun with normalization

Base-Test(1) = image with opened eyes; should be the same as image with closed eyes, as query image
Base-Test(2) = hold, should be different from query image

                        Perfect, Base-Test(1), Base-Test(2) :
Method: 0 Correlation   1.0 / 0.9907733746905749 / 0.9200271749169306                # так же
Method: 1 Chi-squared   0.0 / 1.313295583843248 / 12.388794853714614                 # Метод резко починился
Method: 2 Intersection  59.35084829479456 / 53.323186706751585 / 35.661173459666315  # более явная разница с test2
Method: 3 Bhattacharyya 0.0 / 0.04741276038589942 / 0.1417796188180784               # так же
"""

"""
fall-in-love with normalization

Base-Test(1) = almost the same image as the query one
Base-Test(2) = movement after the hold, so different from the query image

                        Perfect, Base-Test(1), Base-Test(2) :
Method: 0 Correlation   1.0 / 0.9841678650016072 / 0.9785676082357284                # так же
Method: 1 Chi-squared   0.0 / 2.460105862805216 / 4.090048125953625                  # very small diff
Method: 2 Intersection  59.16795280337101 / 51.52063707727939 / 47.848447997821495   # более явная разница с test2
Method: 3 Bhattacharyya 0.0 / 0.06720620614614999 / 0.09677062298839467              # так же
"""

# С нормализацией изображения в черно-белое Intersection лучше всего описывает мои данные.
# chi^2 починился, но он не показывает большую разницу между М и Н.
# Corr & Bhtchrya не меняются почти вообще.
# Intersection + normalization is worth a try