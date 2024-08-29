import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

# read image
image_path = '/home/phillip/Desktop/todays_tutorial/30_text_detection_easyocr/code/data/test2.png'

img = cv2.imread(image_path)

# instance text detector
reader = easyocr.Reader(['en'], gpu=False)

# detect text on image
text_ = reader.readtext(img)

threshold = 0.25
# draw bbox and text
for t_, t in enumerate(text_):
    print(t)

    bbox, text, score = t

    if score > threshold:
        # bbox'un ilk ve son noktalarını kullanarak dikdörtgen çiziyoruz
        start_point = tuple(map(int, bbox[0]))  # Sol üst köşe
        end_point = tuple(map(int, bbox[2]))    # Sağ alt köşe
        cv2.rectangle(img, start_point, end_point, (0, 255, 0), 5)
        
        # Metni yerleştiriyoruz
        cv2.putText(img, text, start_point, cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

plt.figure(figsize=(12, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()