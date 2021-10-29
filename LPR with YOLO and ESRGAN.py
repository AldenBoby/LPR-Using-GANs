#Big thanks to AlexeyAB for his darknet and YOLO framework, likewise xinntao for his ESRGAN code
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import glob
import torch
import sys
import RRDBNet_arch as arch

#-------------------------YOLOv3---------------------------------
def load_model(weights,config):
  '''load a .weights file with OpenCVs deep learning module'''
   net = cv2.dnn.readNet(weights,config)
   classes = ['licence plate'] # as there is only one class for the model it can be defined here without loading a file
   output_layer_names = net.getUnconnectedOutLayersNames() # get the output layers of the model
   return net,output_layer_names

def load_image(path): # load in the image for object detection
   input_img = cv2.imread(path)
   height, width, _ = input_img.shape
   return input_img, height, width

def detection(img, height,width, net, output_layers):
   blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False) # 320 x 320 - higher detection speed, trades some accuracy
   net.setInput(blob)
   outputs = net.forward(output_layers) # retrieve info on the detected objects
   
   boxes = []
   confidences = []
   class_ids = []

   for output in outputs:
         for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores) 
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
   return boxes, confidences


def draw_bbox(boxes, confs, img): 
  indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
  for i in range(len(boxes)):
                if i in indexes:
                  x, y, w, h = boxes[i]
                  label = 'licence plate'
                  predicted_bbox = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2) # draw the bounding box (in red)
  if len(indexes)<=0: # the model was unable to predict a bounding box
      print("The system failed to find a licence plate in the supplied image")
      return []
  else:
    return predicted_bbox # ROI successfully located

image_path = 'Test.png' # insert the path of the image you want to detect here
net, output_layers = load_model('Models/yolov3-train_best.weights','yolov3-train.cfg')
input_img, height, width = load_image(image_path)
boxes, confidences = detection(input_img, height,width,net,output_layers)
labelled_ROI = draw_bbox(boxes, confidences, input_img)


if len(labelled_ROI) != 0: # plot the input image with a marked bounding box
  plt.figure(figsize=(24, 24))
  plt.imshow(cv2.cvtColor(labelled_ROI, cv2.COLOR_BGR2RGB))
  plt.savefig('detected.jpg')
  plt.show()
  
  # crop using the predicted bounding box coordinates to get the ROI
  newimg = Image.open(image_path) 
  x = boxes[0][0]
  y = boxes[0][1]
  w = boxes[0][2]
  h = boxes[0][3]
  crop_img = newimg.crop((x,y,x+w,y+h))
  crop_img.save('cropped.png')
  plt.imshow(crop_img)
  plt.show()

#-------------------------ESRGAN---------------------------------
#Repository for ESRGAN --> github.com/xinntao/ESRGAN
model_path = 'Models/RRDB_ESRGAN_x4.pth' # load the ESRGAN model
device = torch.device('cuda') # switch to cuda for GPU, cpu for CPU

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

img = cv2.imread('cropped.png', cv2.IMREAD_COLOR) # the path of the image to be upscaled goes here
img = img * 1.0 / 255
img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
img_lr = img.unsqueeze(0)
img_lr = img_lr.to(device)

with torch.no_grad():
    output = model(img_lr).data.squeeze().float().cpu().clamp_(0, 1).numpy()
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
output = (output * 255.0).round()
cv2.imwrite('/content/{:s}_hr.png'.format('cropped'), output)

#-------------------------Pytesseract---------------------------------
ROI_HR = cv2.imread('cropped_hr.png') # HR image to be sent to the OCR
ROI_LR = cv2.imread('cropped.png') # LR image to be sent to the OCR

def OCR_preprocess(image):
  ROI_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # greyscale the image
  denoised = cv2.fastNlMeansDenoising(ROI_greyscale,20,20,7,21) # remove noise from the image
  #ret,ROI_thresh = cv2.threshold(denoised,90,255,cv2.THRESH_BINARY)
  #cv2.imwrite("thresh.jpg",ROI_thresh)
  #cv2.imwrite("blurred.jpg",denoised)
  text = pytesseract.image_to_string(denoised,config ='--psm 8') # convert the text in the image to a string, ps8 single line of text
  for char in text:
    if char in '(){}[]abcdedfghijklmnopqrstuvwxyz®_|™~‘":!;,“.?°/@=»>§<¥':
      text = text.replace(char,'') # remove illegal characters from the prediction
  return text

print("ESRGAN (HR)  : ",OCR_preprocess(ROI_HR))
print("ORIGINAL (LR):",OCR_preprocess(ROI_HR))