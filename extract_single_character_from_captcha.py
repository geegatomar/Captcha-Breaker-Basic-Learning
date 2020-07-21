from preprocessor import Preprocessor
import numpy as np
import imutils
import cv2
import os

CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
OUTPUT_FOLDER = "extracted_letter_images"

captcha_image_files = os.listdir(CAPTCHA_IMAGE_FOLDER)

counts = {}

for (i, captcha_image_file) in enumerate(captcha_image_files):
    
    print("[INFO] Processing captcha image {} / {}".format(i + 1, len(captcha_image_files)))

    captcha_correct_text = captcha_image_file.split('.')[0]
    filePath = os.path.join(CAPTCHA_IMAGE_FOLDER, captcha_image_file)
    image = cv2.imread(filePath)
    
    letter_image_regions = []
    gray, contours = Preprocessor.find_contours(image)
    
    for contour in contours:        
        (x, y, w, h) = cv2.boundingRect(contour)

        if w/h > 1.25:
            half_width = int(w/2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))
        
    if len(letter_image_regions) != 4:
        continue
    
    letter_image_regions = sorted(letter_image_regions, key = lambda x : x[0])
    
    
    for (letter_bounding_box, letter_text) in zip(letter_image_regions, captcha_correct_text):        
        (x, y, w, h) = letter_bounding_box
        letter_image = gray[y - 2 : y + h + 2, x - 2 : x + w + 2]
        
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        counts[letter_text] = count + 1


