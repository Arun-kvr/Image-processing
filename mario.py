import cv2
import numpy as np

img_rgb = cv2.imread('mario.png')
if img_rgb is None:
    print("Error: Could not read 'mario.jpg'")
    exit()

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('mariocoin.png', 0)
if template is None:
    print("Error: Could not read 'mariocoin.jpg'")
    exit()

w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

output_file = 'res.png'
cv2.imwrite(output_file, img_rgb)
print(f"Result saved as '{output_file}'")

cv2.imshow("Detected Matches", img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
