from flask import Flask,request,render_template

import cv2
import numpy as np


app = Flask(__name__)

def detect_damage_by_percentage(image):
  # Preprocess the image
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (5, 5), 0)

  # Apply thresholding to detect damaged regions
  _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

  # Find contours of damaged regions
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  damage_image = cv2.drawContours(image.copy(), contours, -1, (0, 0, 255), 2)

  cv2.imshow('Damage Detection', damage_image)

  # Calculate the total area of the damaged regions
  total_area = sum(cv2.contourArea(contour) for contour in contours)

  # Calculate the percentage of damage
  image_area = image.shape[0] * image.shape[1]
  damage_percentage = (total_area / image_area) * 100

  # cv2.waitKey(0)
  cv2.destroyAllWindows()

  return damage_percentage

@app.route('/')
def index():
  return render_template('index.html')


@app.route('/damage_request',methods=['Post'])
def damage_request():
  image_file = request.files['image']
  # cv2.imshow(image_file)
  image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

  # Calculate the damage percentage
  percentage = detect_damage_by_percentage(image)

  return f"{percentage:.2f}%"


if __name__ == '__main__':
  app.run(debug=True)