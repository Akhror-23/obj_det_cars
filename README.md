# obj_det_cars
#obj_det_cars


from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# Load YOLOv8 object detection model
model_det = YOLO("yolov8n.pt")  # You can use yolov8s.pt or yolov8m.pt for more accuracy

# Run detection on your image
image_path = "C:/Users/International/OneDrive/Рабочий стол/test_images_cars/image2.jpg"  # Use forward slashes to avoid escape errors
results_det = model_det(image_path)[0]

# Show detection results with bounding boxes
results_det.show()

# Save output image in the same directory as the input image
input_dir = os.path.dirname(image_path)
output_path = os.path.join(input_dir, "image2_detected.jpg")
results_det.save(filename=output_path)

print(f"Detection completed and result saved at: {output_path}")
