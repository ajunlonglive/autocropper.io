import os
import random
import string
from flask import Flask, redirect, render_template, request, send_from_directory, url_for
from flask import make_response
import cv2
from PIL import Image as im
import numpy as np
import base64
import imutils 
from skimage.filters import threshold_local
import json

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
DOWNLOAD_FOLDER = os.path.basename('downloads')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

@app.route("/")
def start_page():
    # print("Start")
    return render_template('index.html')

# @app.route('/uploader', methods=['GET', 'POST'])
# def uploader():
#     if request.method == 'POST':
#         # do stuff when the form is submitted

#         # redirect to end the POST handling
#         # the redirect can be to the same route or somewhere else
#         return redirect(url_for('index'))

#     # show the form, it wasn't submitted
#     return render_template('uploader.html')

@app.route('/uploads/<path:filename>')
def download_file(filename):
    # print("pizzaaa")
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename, as_attachment=False)

@app.route('/getscans', methods=['POST'])
def getscans():
    file = request.files['image']

    other = request.args
    print(other)

    for key in other:
        print(other.get(key))
        print(other.getlist(key))

    # Read image
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Detect faces
    faces = detect_faces(image)

    if len(faces) == 0:
        faceDetected = False
        num_faces = 0
        to_send = ''
    else:
        faceDetected = True
        num_faces = len(faces)

        # In memory
        images_to_send = []
        images_filepaths = []
        count = 0

        # for json response
        encoded_images = []

        for box_coords in faces:
            count+=1
            (x, y, w, h) = box_coords
            cropped = image[y:y+h, x:x+w]

            fancy_autocropped = autocrop(cropped)

            # images_filepaths.append(save_image_to_disk(cropped, count))
            images_filepaths.append(save_image_to_disk(fancy_autocropped, count))


            # image_content = cv2.imencode('.jpg', cropped)[1].tostring()
            image_content = cv2.imencode('.jpg', fancy_autocropped)[1].tostring()
            encoded_image = base64.b64encode(image_content)
            encoded_images.append(encoded_image)


            to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
            # https://base64.guru/converter/decode/image/png
            # this actually works using Postman if you remove the data:image/jpg;base64 bullshit

            images_to_send.append(to_send)

    response = {
        "images": encoded_images,
        "age": 30,
        "city": "New York"
    }

    return response

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']

    other = request.args
    print(other)

    for key in other:
        print(other.get(key))
        print(other.getlist(key))

    # Read image
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Detect faces
    faces = detect_faces(image)

    if len(faces) == 0:
        faceDetected = False
        num_faces = 0
        to_send = ''
    else:
        faceDetected = True
        num_faces = len(faces)

        # In memory
        images_to_send = []
        images_filepaths = []
        count = 0
        for box_coords in faces:
            count+=1
            (x, y, w, h) = box_coords
            cropped = image[y:y+h, x:x+w]

            fancy_autocropped = autocrop(cropped)

            # images_filepaths.append(save_image_to_disk(cropped, count))
            images_filepaths.append(save_image_to_disk(fancy_autocropped, count))


            # image_content = cv2.imencode('.jpg', cropped)[1].tostring()
            image_content = cv2.imencode('.jpg', fancy_autocropped)[1].tostring()
            encoded_image = base64.b64encode(image_content)

            to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
            # https://base64.guru/converter/decode/image/png
            # this actually works using Postman if you remove the data:image/jpg;base64 bullshit

            images_to_send.append(to_send)

    return render_template('index.html', faceDetected=faceDetected, 
                                         num_faces=num_faces, 
                                         image_to_show=images_to_send, 
                                         init=True,
                                         filenames = images_filepaths)

def save_image_to_disk(img, i):
    test = im.fromarray(img)
    filename = "uploads/myimage" + str(i) + ".png"
    test.save(filename)
    return filename
# ----------------------------------------------------------------------------------
# Detect faces using OpenCV
# ----------------------------------------------------------------------------------  
def detect_faces(img):
    '''Detect face in an image'''
    image1 = img
    h, w = image1.shape[0:2]

    # Make border for easier crop
    image = cv2.copyMakeBorder(
        image1,
        30,
        30,
        30,
        30,
        cv2.BORDER_CONSTANT,
        value=(255,255,255)
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    ret, th = cv2.threshold(gray, 210, 235, 1)

    cnts, hierarchy = cv2.findContours(th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    best_contours = list()

    for c in cnts:
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        Area = image.shape[0] * image.shape[1]
        if Area / 10 < cv2.contourArea(box) < Area * 2 / 3:
            best_contours.append(box)

    faces_list = []

    for c in best_contours:
        # get the bounding rect
        (x, y, w, h) = cv2.boundingRect(c)

        # faces_list.append(image[y:y + h, x:x + w])
        faces_list.append(cv2.boundingRect(c))

    # Return the face image area and the face rectangle
    return faces_list


# New methods for final cropping
def order_rect(points):
    # idea: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    # initialize result -> rectangle coordinates (4 corners, 2 coordinates (x,y))
    res = np.zeros((4, 2), dtype=np.float32)    

    # top-left corner: smallest sum
    # top-right corner: smallest difference
    # bottom-right corner: largest sum
    # bottom-left corner: largest difference

    s = np.sum(points, axis = 1)    
    d = np.diff(points, axis = 1)

    res[0] = points[np.argmin(s)]
    res[1] = points[np.argmin(d)]
    res[2] = points[np.argmax(s)]
    res[3] = points[np.argmax(d)]

    return res

def four_point_transform(img, points):    
    # copied from: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_rect(points)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = np.float32)

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped

def cont(img, gray, user_thresh, crop):
    # CONTINUE
    found = False
    loop = False
    old_val = 0 # thresh value from 2 iterations ago
    i = 0 # number of iterations

    im_h, im_w = img.shape[:2]
    while found == False: # repeat to find the right threshold value for finding a rectangle
        if user_thresh >= 255 or user_thresh == 0 or loop: # maximum threshold value, minimum threshold value 
                                                 # or loop detected (alternating between 2 threshold values 
                                                 # without finding borders            
            break # stop if no borders could be detected

        ret, thresh = cv2.threshold(gray, user_thresh, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]        
        im_area = im_w * im_h
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (im_area/100) and area < (im_area/1.01):
                epsilon = 0.1 * cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)

                if len(approx) == 4:
                    found = True
                elif len(approx) > 4:
                    user_thresh = user_thresh - 1
                    print(f"Adjust Threshold: {user_thresh}")
                    if user_thresh == old_val + 1:
                        loop = True
                    break
                elif len(approx) < 4:
                    user_thresh = user_thresh + 5
                    print(f"Adjust Threshold: {user_thresh}")
                    if user_thresh == old_val - 5:
                        loop = True
                    break

                rect = np.zeros((4, 2), dtype = np.float32)
                rect[0] = approx[0]
                rect[1] = approx[1]
                rect[2] = approx[2]
                rect[3] = approx[3]
                
                dst = four_point_transform(img, rect)
                dst_h, dst_w = dst.shape[:2]
                img = dst[crop:dst_h-crop, crop:dst_w-crop]
            else:
                if i > 100:
                    # if this happens a lot, increase the threshold, maybe it helps, otherwise just stop
                    user_thresh = user_thresh + 5
                    if user_thresh > 255:
                        break
                    print(f"Adjust Threshold: {user_thresh}")
                    print("WARNING: This seems to be an edge case. If the result isn't satisfying try lowering the threshold using -t")
                    if user_thresh == old_val - 5:
                        loop = True                
        i += 1
        if i%2 == 0:
            old_value = user_thresh

    return found, img

def deskew(im, max_skew=1):
    # height, width = im.shape
    height,width = im.shape[:2]

    # Create a grayscale image and denoise it
    im_gs = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gs = cv2.fastNlMeansDenoising(im_gs, h=3)

    # Create an inverted B&W copy using Otsu (automatic) thresholding
    im_bw = cv2.threshold(im_gs, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Detect lines in this image. Parameters here mostly arrived at by trial and error.
    lines = cv2.HoughLinesP(
        im_bw, 1, np.pi / 180, 200, minLineLength=width / 12, maxLineGap=width / 150
    )

    # Collect the angles of these lines (in radians)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angles.append(np.arctan2(y2 - y1, x2 - x1))

    # If the majority of our lines are vertical, this is probably a landscape image
    landscape = np.sum([abs(angle) > np.pi / 4 for angle in angles]) > len(angles) / 2

    # Filter the angles to remove outliers based on max_skew
    if landscape:
        angles = [
            angle
            for angle in angles
            if np.deg2rad(90 - max_skew) < abs(angle) < np.deg2rad(90 + max_skew)
        ]
    else:
        angles = [angle for angle in angles if abs(angle) < np.deg2rad(max_skew)]

    if len(angles) < 5:
        # Insufficient data to deskew
        return im

    # Average the angles to a degree offset
    angle_deg = np.rad2deg(np.median(angles))

    # If this is landscape image, rotate the entire canvas appropriately
    if landscape:
        if angle_deg < 0:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            angle_deg += 90
        elif angle_deg > 0:
            im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
            angle_deg -= 90

    # Rotate the image by the residual offset
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1)
    im = cv2.warpAffine(im, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return im

def autocrop(input_img):
    # img = cv2.imread(input_img)
    img = input_img

    thresh = 210
    crop = 1

    #add white background (in case one side is cropped right already, otherwise script would fail finding contours)
    img = cv2.copyMakeBorder(img,100,100,100,100, cv2.BORDER_CONSTANT,value=[255,255,255])
    im_h, im_w = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    res_gray = cv2.resize(img,(int(im_w/6), int(im_h/6)), interpolation = cv2.INTER_CUBIC)
    found, img = cont(img, gray, thresh, crop)

    if found:
        print("IMAGE FOUND")
        # rotated = deskew(img)
        return img
    else:
        # if no contours were found, write input file to "failed" folder
        print("No image found in 2nd autocrop method, returning input")
        return input_img


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=80)