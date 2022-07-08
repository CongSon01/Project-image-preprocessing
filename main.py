from re import L
from tkinter import *
from tkinter import filedialog
import os
import tkinter as tk
from PIL import Image, ImageTk # pip install Pillow
# import RGB2HSI
import numpy as np
import math
import cv2
from tkinter import messagebox

import pre_processing
import script

img_obj = ''
img_rgb, img_cmy, img_hsi = '', '', ''
is_rgb, is_cmy, is_hsi = True, False, False

def changeimage(img_change):
    img = ImageTk.PhotoImage(img_change)
    lbl.configure(image=img)
    lbl.image = img

def change_gray_new():
    global img_obj
    # convert to numpy
    img_arr = np.array(img_obj)
    img_gray = pre_processing.convert_gray(img_arr)
    img_gray = Image.fromarray(img_gray)
    changeimage(img_gray)
    print('Hello Gray')

def showimage():
    global img_obj
    fln = filedialog.askopenfilename(initialdir = os.getcwd(),title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")))
    img_obj = Image.open(fln)

    changeimage(img_obj)

def change_GRAY():
    global img_obj
    img_gray = img_obj.convert("L")
    changeimage(img_gray)

def change_title():
    root.title("Dương Công Sơn")

def login():
    top = Toplevel()
    top.title("Đổi title")
    #Set the geometry of tkinter frame
    top.geometry("200x200")
    # Create text widget and specify size.
    T = Text(top, height = 5, width = 52)
    # Create label
    l = Label(top, text = "Nhập title:")
    # Create button
    b = Button(top, text = "OK", command = change_title)
    l.pack()
    T.pack()
    b.pack()
    top.mainloop()

    
root = Tk()

frm = Frame(root)
frm.pack(side=BOTTOM, padx=10, pady=10)

lbl = Label(root)
lbl.pack()

def run_knn():
    image = cv2.imread('digits.png')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # show image gray
    small = cv2.pyrDown(image)
    
    cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

    x = np.array(cells)
    print ("The shape of our cells array: " + str(x.shape))

    train = x[:,:70].reshape(-1,400).astype(np.float32) 
    test = x[:,70:100].reshape(-1,400).astype(np.float32) 


    k = [0,1,2,3,4,5,6,7,8,9]
    train_labels = np.repeat(k,350)[:,np.newaxis]
    test_labels = np.repeat(k,150)[:,np.newaxis]

    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    ret, result, neighbors, distance = knn.findNearest(test, k=3)

    matches = result == test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct * (100.0 / result.size)
    print("Accuracy is = %.2f" % accuracy + "%")
    return knn

def change_OTSU():
    global img_obj
    # convert to numpy array
    img_arr = np.array(img_obj)
    # convert OTSU thresholding
    thread_otsu = pre_processing.convert_otsu(img_arr)
    # convert to Image object
    img_otsu = Image.fromarray(thread_otsu)
    # update image
    changeimage(img_otsu)

def change_Adaptive():
    global img_obj
    # convert to numpy array
    img_arr = np.array(img_obj)
    # convert gray
    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    # convert Adaptive thresholding using cv2
    thread_adaptive = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 15)
    # convert to Image object
    img_adaptive = Image.fromarray(thread_adaptive)
    # update image
    changeimage(img_adaptive)

def change_erosion():
    global img_obj
    # convert to numpy array
    img_arr = np.array(img_obj)
    kernel = np.ones((3,3),np.uint8)
    # convert otsu
    thread_otsu = pre_processing.convert_otsu(img_arr)
    # convert erosion
    thread_erosion = cv2.erode(thread_otsu,kernel,iterations = 1)
    # convert to Image object
    img_erosion = Image.fromarray(thread_erosion)
    # update image
    changeimage(img_erosion)

def change_dilation():
    global img_obj
    # convert to numpy array
    img_arr = np.array(img_obj)
    kernel = np.ones((3,3),np.uint8)
    # convert otsu
    thread_otsu = pre_processing.convert_otsu(img_arr)
    # convert dilation
    thread_dilation = cv2.dilate(thread_otsu,kernel,iterations = 1)
    # convert to Image object
    img_dilation = Image.fromarray(thread_dilation)
    # update image
    changeimage(img_dilation)

def change_mean_filter():
    global img_obj
    # convert to numpy array
    img_arr = np.array(img_obj)
    denoised=cv2.fastNlMeansDenoising(img_arr,None,10,10)
    # convert to Image object
    img_denoise = Image.fromarray(denoised)
    # update image
    changeimage(img_denoise)

def change_Gaussian():
    global img_obj
    # convert to numpy array
    img_arr = np.array(img_obj)
    denoised=cv2.GaussianBlur(img_arr,(3,3),0)
    # convert to Image object
    img_denoise = Image.fromarray(denoised)
    # update image
    changeimage(img_denoise)

def change_median_filter():
    global img_obj
    # convert to numpy array
    img_arr = np.array(img_obj)
    denoised=cv2.medianBlur(img_arr,3)
    # convert to Image object
    img_closing = Image.fromarray(denoised)
    # update image
    changeimage(img_closing)

def change_Opening():
    global img_obj
    # convert to numpy array
    img_arr = np.array(img_obj)
    kernel = np.ones((3,3),np.uint8)
    # convert otsu
    thread_otsu = pre_processing.convert_otsu(img_arr)
    # convert opening
    thread_opening = cv2.morphologyEx(thread_otsu, cv2.MORPH_OPEN, kernel)
    # convert to Image object
    img_opening = Image.fromarray(thread_opening)
    # update image
    changeimage(img_opening)

def change_Closing():
    global img_obj
    # convert to numpy array
    img_arr = np.array(img_obj)
    kernel = np.ones((3,3),np.uint8)
    # convert otsu
    thread_otsu = pre_processing.convert_otsu(img_arr)
    # convert closing
    thread_closing = cv2.morphologyEx(thread_otsu, cv2.MORPH_CLOSE, kernel)
    # convert to Image object
    img_closing = Image.fromarray(thread_closing)
    # update image
    changeimage(img_closing)

def change_fourier():
    pass

def change_canny():
    global img_obj
    # convert to numpy array
    img_arr = np.array(img_obj)
    # convert gray
    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    # convert canny
    thread_canny = cv2.Canny(img_gray, 100, 250)
    # convert to Image object
    img_canny = Image.fromarray(thread_canny)
    # update image
    changeimage(img_canny)

def change_contour():
    global img_obj
    # convert to numpy array
    img_arr = np.array(img_obj)
    # convert gray
    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    thread_canny = cv2.Canny(img_gray, 100, 200)
    contours, _ = cv2.findContours(thread_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    full_number = []

    # loop over the contours
    for c in contours:
        # compute the bounding box for the rectangle
        (x, y, w, h) = cv2.boundingRect(c)    
        
        #cv2.drawContours(image, contours, -1, (0,255,0), 3)
        #cv2.imshow("Contours", image)

        if w >= 5 and h >= 25:
            cv2.rectangle(img_arr, (x, y), (x + w, y + h), (0, 0, 255), 2)

    im_pil = Image.fromarray(img_arr)
    changeimage(im_pil)

def change_contour_done():
    global img_obj
    # convert to numpy array
    img_arr = np.array(img_obj)
    thread_otsu = pre_processing.convert_otsu(img_arr)
    # img_cmy = predict(img_obj)
    gray = cv2.cvtColor(thread_otsu,cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thread_canny = cv2.Canny(blurred, 100, 200)
    contours, _ = cv2.findContours(thread_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)  
        if w >= 5 and h >= 25:
            cv2.rectangle(img_arr, (x, y), (x + w, y + h), (0, 0, 255), 2)

    im_pil = Image.fromarray(img_arr)
    changeimage(im_pil)

def change_binary():
    global img_obj
    # convert to numpy array
    img_arr = np.array(img_obj)
    # convert gray
    ret, roi = cv2.threshold(img_arr, 127, 255,cv2.THRESH_BINARY_INV)
    # convert to Image object
    img_binary = Image.fromarray(roi)
    # update image
    changeimage(img_binary)

def click_predict():
    global img_obj, is_rgb, is_hsi, is_cmy, img_cmy
    knn = run_knn()
    # print(get_value_image(img_obj))
    # convert Image object to numpy array
    new_image = img_obj.convert('RGB') # convert to RGB
    new_image = np.array(new_image) # convert to numpy array
    new_image = new_image[:, :, ::-1].copy() 
    thread_otsu = pre_processing.convert_otsu(new_image)
    print(new_image.shape)
    # img_cmy = predict(img_obj)
    gray = cv2.cvtColor(thread_otsu,cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 30, 150)
    cv2.imshow("edged", edged)
    # print('CMY: ',get_value_image(img_cmy))
    # Fint Contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    full_number = []

    # loop over the contours
    for c in contours:
        # compute the bounding box for the rectangle
        (x, y, w, h) = cv2.boundingRect(c)    
        
        #cv2.drawContours(image, contours, -1, (0,255,0), 3)
        #cv2.imshow("Contours", image)

        if w >= 5 and h >= 25:
            roi = blurred[y:y + h, x:x + w]
            ret, roi = cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)
            squared = script.makeSquare(roi)
            final = script.resize_to_pixel(20, squared)
            # cv2.imshow("final", final)
            final_array = final.reshape((1,400))
            final_array = final_array.astype(np.float32)
            ret, result, neighbours, dist = knn.findNearest(final_array, k=1)
            number = str(int(float(result[0])))
            full_number.append(number)
            # draw a rectangle around the digit, the show what the
            # digit was classified as
            cv2.rectangle(new_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(new_image, number, (x , y + 155),
                cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)

    im_pil = Image.fromarray(new_image)
    changeimage(im_pil)
    # print('Predict')

# Add Image
image_upload = Image.open("upload.jpg")
resize_image = image_upload.resize((60, 70))
upload_btn = ImageTk.PhotoImage(resize_image)
btn = Button(frm, image=upload_btn , command=showimage)
btn.grid(row=1, rowspan=2, column=0, padx=15)

image_upload_1 = Image.open("login.png")
resize_image_1 = image_upload_1.resize((70, 20))
upload_btn_1 = ImageTk.PhotoImage(resize_image_1)
btn_login = Button(frm, image=upload_btn_1 , command=login)
btn_login.grid(row=3, rowspan=1, column=0, padx=15)


btn1 = Button(frm, text="Gray", command=change_gray_new)
btn1.grid(row=0, column=1, padx=10)

btn2 = Button(frm, text="OTSU", command=change_OTSU)
btn2.grid(row=0, column=2, padx=10)

btn3 = Button(frm, text="Adaptive", command=change_Adaptive)
btn3.grid(row=0, column=3, padx=10)

btn4 = Button(frm, text="Predict", command=click_predict)
btn4.grid(row=0, column=4, padx=10)

btn5 = Button(frm, text="Exit", command=lambda: exit())
btn5.grid(row=0, column=5, padx=10)

btn6 = Button(frm, text="Dilation", command=change_erosion)
btn6.grid(row=1, column=1, padx=10)

btn7 = Button(frm, text="Erosion", command=change_dilation)
btn7.grid(row=1, column=2, padx=10)

btn12 = Button(frm, text="Opening", command=change_Opening)
btn12.grid(row=1, column=3, padx=10)

btn13 = Button(frm, text="Closing", command=change_Closing)
btn13.grid(row=1, column=4, padx=10)

btn8 = Button(frm, text="median filter", command=change_mean_filter)
btn8.grid(row=2, column=1, padx=10)

btn9 = Button(frm, text="mean filter", command=change_median_filter)
btn9.grid(row=2, column=2, padx=10)

btn10 = Button(frm, text="gaussian filter", command=change_Gaussian)
btn10.grid(row=2, column=3, padx=10)

btn11 = Button(frm, text="fourier filter", command=change_fourier)
btn11.grid(row=2, column=4, padx=10)

btn14 = Button(frm, text="Canny", command=change_canny)
btn14.grid(row=3, column=2, padx=10)

btn15 = Button(frm, text="Contour", command=change_contour)
btn15.grid(row=3, column=3, padx=10)

btn16 = Button(frm, text="Contour_done", command=change_contour_done)
btn16.grid(row=3, column=4, padx=10)

btn17 = Button(frm, text="Binary", command=change_binary)
btn17.grid(row=3, column=1, padx=10)

root.title("Nhóm 3 - 64CS3")
#Set the geometry of tkinter frame
root.geometry("600x700")
root.mainloop()
