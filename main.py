import os
from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
import imutils 
from skimage.filters import threshold_local

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def start_page():
    print("Start")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']

    # Save file
    #filename = 'static/' + file.filename
    #file.save(filename)

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
        
        # Draw a rectangle - commented out because no need
        # for box_coords in faces:
        #     draw_rectangle(image, box_coords)
        
        # Save
        # cv2.imwrite("filename", image)
        
        images = []
        # In memory
        images_to_send = []
        for box_coords in faces:
            # (x, y), (x + w, y + h)
            (x, y, w, h) = box_coords
            # cropped = image[x:y, (x+w):(y+h)]
            cropped = image[y:y+h, x:x+w]
            image_content = cv2.imencode('.jpg', cropped)[1].tostring()
            encoded_image = base64.b64encode(image_content)
            to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
            images_to_send.append(to_send)

    return render_template('index.html', faceDetected=faceDetected, num_faces=num_faces, image_to_show=images_to_send, init=True)

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
# ----------------------------------------------------------------------------------
# Draw rectangle on image
# according to given (x, y) coordinates and given width and heigh
# ----------------------------------------------------------------------------------
def draw_rectangle(img, rect):
    '''Draw a rectangle on the image'''
    (x, y, w, h) = rect
    x = x-30
    y = y-30

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=80)