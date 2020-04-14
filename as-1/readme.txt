ASSIGNMENT -1

Question 1) Hybrid Images

    This question is about forming a hybrid image using two input images. Low frequency information is extracted from one input image by convolving with Gaussian Filter.
High frequency information is obtained from second image by subtracting its low frequency information from it which is again extracted using Gaussian Filter. A hybrid image
formed by combining the above two images(combining the frequency information obtained from both). Cutoff frequencies are varied in terms of sigma(standard deviation).Best 
results are obtained when cutoff frequencies are taken as alpha = 8, beta = 8 and kernal size as sigma*6.

Question 3) Space-Scale Blob Detection

    In this question, gaussian filter is double differentiated and normalized to form a Blob filter of size six sigma. Then, a for loop is implemented in which sigma is updated 
by a value of k power i(loop variable). Image and filter are convoluted in this loop and the result is squared and stored. Now maximum values are obtained from sliced matrices of the 
images and the coordinates are stored as blobs. These blobs are displayed in the image and to decrease the redundancy, minimum distance between every two blobs is calculated and blobs with
less than a threshold distance are removed and more than that distance are displayed.

Question 2) Harris corner Detection,shi-thomasi corner detection

   In this algorithm, we need to find the difference in intensity of displacement in all directions. Window function generally gives weights to the pixels. We set the window size as 10. After calculating
the intensity, we need to differentiate with sobel kernel in X and Y directions respectively. Finally by using all these values, we calculate the Harris response. Then a threshold is set as 14 such that
if R value is greater than threshold, we detect it as a corner these dots will be represented in blue dots . in shi-thomasi corner detection algorithm we calculated eigen values then we should consider the 
minimum value of eigen values.if that minimumvalue is greater than threshold which is set to 4 then it is considered as a corner of the image these dots will be represented in blue dots and for both algorithms
 alpha is set to 0.04
  