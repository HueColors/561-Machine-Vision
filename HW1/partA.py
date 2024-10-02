import cv2

# Lists to store coordinates
points_img1 = []
points_img2 = []

# Mouse callback function to capture clicks and display points
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append point to the respective image list
        param[0].append((x, y))
        # Draw a small circle at the clicked point
        cv2.circle(param[1], (x, y), 5, (0, 0, 255), -1)  # Red color with filled circle
        # Update the display with the point drawn
        cv2.imshow(param[2], param[1])
        print(f"Point selected on {param[2]}: {(x, y)}")

# Load the first image
img1 = cv2.imread(r'C:\Users\hnguy\Desktop\img1.jpg')

# Resize the image by 30%
scale_percent = 30
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
img1 = cv2.resize(img1, dim)

# Load the second image
img2 = cv2.imread(r'C:\Users\hnguy\Desktop\img2.jpg')

# Resize the second image as well
img2 = cv2.resize(img2, dim)

# Show both images side by side
cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)

# Set mouse callbacks for both images
cv2.setMouseCallback('Image 1', select_points, param=[points_img1, img1, 'Image 1'])
cv2.setMouseCallback('Image 2', select_points, param=[points_img2, img2, 'Image 2'])

print("Click on 8 points in both images, press 'q' when done")

# Keep both windows open until points are selected
while True:
    if cv2.waitKey(1) & 0xFF == ord('q') and (len(points_img1) >= 8 and len(points_img2) >= 8):
        break

# Save the images with the selected points
cv2.imwrite(r'C:\Users\hnguy\Desktop\img1_with_points.jpg', img1)
cv2.imwrite(r'C:\Users\hnguy\Desktop\img2_with_points.jpg', img2)
print("Images with selected points have been saved!")

cv2.destroyAllWindows()

# Print out the selected points
print("Selected points in Image 1: ", points_img1)
print("Selected points in Image 2: ", points_img2)
