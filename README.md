# Captcha Breaker Basic Learning
Simple Captcha Breaker which has 99.8% accuracy on 4 letter WordPress Captchas.

Note: The images in this dataset are very small, hence the correct preprocessing would be:
```
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[1] if imutils.is_cv3() else contours[0]
        
 ```
 Dont do this:
        #Blurring and canny is a bad idea for this question as the images are too small. So just do thresh.
        #blurred = cv2.GaussianBlur(image, (3, 3), 0)
        #edged = cv2.Canny(blurred, 30, 180)
