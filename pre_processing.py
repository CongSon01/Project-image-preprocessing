
import numpy as np
import matplotlib.pyplot as plt

def convert_gray(img):
    """
    Convert image to gray scale
    """
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return gray

def convert_otsu(image):
    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]

    print("Otsu threshold: ", threshold)

    image[image<threshold] = 0
    image[image>threshold] = 255
    return image

def threshold_adaptive(image, block_size=3, method='mean', C=0):
    block_size = (block_size,) * image.ndim
    block_size = tuple(block_size)
    image = image.astype('float64', copy=False)
    thresh_image = np.zeros(image.shape, dtype='float64')
    if method == 'mean':
        # calculate the mean in each block 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                block = image[i:i+block_size[0], j:j+block_size[1]]
                thresh_image[i,j] = block.mean()
    elif method == 'median':
        # calculate the median in each block 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                block = image[i:i+block_size[0], j:j+block_size[1]]
                thresh_image[i,j] = np.median(block)
    elif method == 'min_max':
	    # calculate the min and max in each block 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                block = image[i:i+block_size[0], j:j+block_size[1]]
                thresh_image[i,j] = (block.max() - block.min()) / 2
    return thresh_image + C

def dilate_this(image_file, dilation_level=3):
    # setting the dilation_level
    dilation_level = 3 if dilation_level < 3 else dilation_level

    # obtain the kernel by the shape of (dilation_level, dilation_level)
    structuring_kernel = np.full(shape=(dilation_level, dilation_level), fill_value=255)
    image_src = convert_otsu(image_file=image_file)

    orig_shape = image_src.shape
    pad_width = dilation_level - 2

    # pad the image with pad_width
    image_pad = np.pad(array=image_src, pad_width=pad_width, mode='constant')
    pimg_shape = image_pad.shape
    h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])

    # obtain the submatrices according to the size of the kernel
    flat_submatrices = np.array([
        image_pad[i:(i + dilation_level), j:(j + dilation_level)]
        for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)
    ])

    # replace the values either 255 or 0 by dilation condition
    image_dilate = np.array([255 if (i == structuring_kernel).any() else 0 for i in flat_submatrices])
    # obtain new matrix whose shape is equal to the original image size
    image_dilate = image_dilate.reshape(orig_shape)

    return image_dilate

def meanFilter(image):
    width = image.shape[1]
    height = image.shape[0]
    result = np.zeros((image.shape[0], image.shape[1]), int)
    for row in range(height):
        for col in range(width):  
            currentElement=0; left=0; right=0; top=0; bottom=0; topLeft=0; 
            topRight=0; bottomLeft=0; bottomRight=0;
            counter = 1           
            currentElement = image[row][col]

            if not col-1 < 0:
                left = image[row][col-1]
                counter +=1                        
            if not col+1 > width-1:
                right = image[row][col+1]
                counter +=1 
            if not row-1 < 0:
                top = image[row-1][col]
                counter +=1 
            if not row+1 > height-1:
                bottom = image[row+1][col]
                counter +=1 

            if not row-1 < 0 and not col-1 < 0:
                topLeft = image[row-1][col-1]
                counter +=1 
            if not row-1 < 0 and not col+1 > width-1:
                topRight = image[row-1][col+1]
                counter +=1 
            if not row+1 > height-1 and not col-1 < 0:
                bottomLeft = image[row+1][col-1]
                counter +=1 
            if not row+1 > height-1 and not col+1 > width-1:
                bottomRight = image[row+1][col+1]
                counter +=1

            total = int(currentElement)+int(left)+int(right)+int(top)+int(bottom)+int(topLeft)+int(topRight)+int(bottomLeft)+int(bottomRight)
            avg = total/counter
            result[row][col] = avg
    return result
