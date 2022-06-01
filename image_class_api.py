#import tensorflow.keras.applications.resnet50 as resnet50
#from tensorflow.keras.preprocessing import image
import os
import io
import cv2
import imutils
import base64
from flask import jsonify
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # so that it runs on a mac
#app.secret_key = "caircocoders-ednalan"
APP_ROOT = os.path.dirname(os.path.abspath(__file__)) 
print(APP_ROOT)
def crop(fname):
    
    #import numpy as np
    target = os.path.join(APP_ROOT, 'outputs')
    print(target)

    #template = cv2.imread('D:/codes/small_image_a.png')
    #template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    #template = cv2.Canny(template, 50, 200)
    #(tH, tW) = template.shape[:2]


    image = cv2.imread(fname)
    model = core.Model.load('D:/codes/outputs/model_weights_new.pth', ['matrix'])
    predictions = model.predict(image)
    labels, boxes, scores = predictions
    filtered_indices=np.where(max(scores))
    filtered_boxes=boxes[filtered_indices]
    num_list = filtered_indices[0].tolist()
    filtered_labels = [labels[i] for i in num_list]
    xmin=filtered_boxes[0,0].item()
    ymin=filtered_boxes[0,1].item()
    xmax=filtered_boxes[0,2].item()
    ymax=filtered_boxes[0,3].item()
    xmin=round(xmin)
    ymin=round(ymin)
    xmax=round(xmax)
    ymax=round(ymax)
    roi = image[ymin:ymax, xmin:xmax]
    #roi=cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    destination = "/".join([target, 'matrix.jpg'])
    cv2.imwrite(destination,roi)
    #show_labeled_image(image, filtered_boxes, filtered_labels)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    #image_no=1
    '''
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        clone = np.dstack([image])
        cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
        roi = clone[maxLoc[0]: maxLoc[1] + tH,maxLoc[1]: maxLoc[0] + tW]
        name = 'roi_' + str(image_no) + '.png'
        image_no=image_no+1
        destination = "/".join([target, name])
        cv2.imwrite(destination, roi)
    destination = "/".join([target, name]) 
    with open(destination, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    '''
    with open(destination, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    destination_1 = "/".join([target, 'encode.txt'])
    with open(destination_1, "wb") as file:
        file.write(encoded_string)
    
    #rawBytes = io.BytesIO()
    #destination.save(rawBytes, "JPEG")
    #rawBytes.seek(0)
    #img_base64 = base64.b64encode(rawBytes.read())
    
    response = {
         "image":encoded_string.decode(),
         "message":"Image is BASE 64 encoded"        
            }

            #cv2.imwrite(destination_1,large_image)
    
    return response

    '''
    #cv2.imshow("Template", template)
    #cv2.waitKey(0)

    """returns top 5 categories for an image.
    
    :param fname : path to the file 
    """
    # ResNet50 is trained on color images with 224x224 pixels
    input_shape = (224, 224, 3)

    # load and resize image ----------------------
    
    img = image.load_img(fname, target_size=input_shape[:2])
    x = image.img_to_array(img)

    # preprocess image ---------------------------

    # make a batch
    import numpy as np
    x = np.expand_dims(x, axis=0)
    print(x.shape)

    # apply the preprocessing function of resnet50
    img_array = resnet50.preprocess_input(x)

    model = resnet50.ResNet50(weights='imagenet',
                              input_shape=input_shape)
    preds = model.predict(x)
    return resnet50.decode_predictions(preds)
'''

if __name__ == '__main__':

    import pprint
    import sys

    file_name = sys.argv[1]
    results = crop(file_name)
    pprint.pprint(results)