import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Initialize YOLO object detection model
model = YOLO('yolov8m.pt')  # Object detection model

# Initialize YOLO segmentation model
seg_model = YOLO('yolov8m-seg.pt')  # Segmentation model

# Define path lines and counts
path1_line_start = (200, 500)
path1_line_end = (650, 500)
path2_line_start = (650, 500)
path2_line_end = (1150, 500)
path3_line_start = (950, 150)
path3_line_end = (1200, 120)
path1_count = 0
path2_count = 0
path3_count = 0

# Initialize sets to track crossed objects
crossed_objects = set()

def draw_paths_and_labels(frame):
    # Draw Path 1 (Green)
    cv2.line(frame, path1_line_start, path1_line_end, (0, 255, 0), 4)
    cv2.putText(frame, 'Path 1', (path1_line_start[0], path1_line_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw Path 2 (Blue)
    cv2.line(frame, path2_line_start, path2_line_end, (255, 0, 0), 4)
    cv2.putText(frame, 'Path 2', (path2_line_start[0], path2_line_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw Path 3 (Red)
    cv2.line(frame, path3_line_start, path3_line_end, (0, 0, 255), 4)
    cv2.putText(frame, 'Path 3', (path3_line_start[0], path3_line_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def get_class_name(class_id):
    """Map COCO class ID to class name."""
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    else:
        return 'Unknown'

def line_intersection(line1_start, line1_end, line2_start, line2_end):
    """Check if two line segments intersect."""

    def orientation(p, q, r):
        """
        Calculate the orientation of the triplet (p, q, r).

        Returns:
            int: 0 if collinear, 1 if clockwise, 2 if counter-clockwise.
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        elif val > 0:
            return 1
        else:
            return 2

    def on_segment(p, q, r):
        """
        Check if point q lies on segment pr.

        Returns:
            bool: True if q lies on segment pr, False otherwise.
        """
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    p1, q1 = line1_start, line1_end
    p2, q2 = line2_start, line2_end

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if (o1 != o2 and o3 != o4) or (o1 == 0 and on_segment(p1, p2, q1)) or (o2 == 0 and on_segment(p1, q2, q1)) or \
            (o3 == 0 and on_segment(p2, p1, q2)) or (o4 == 0 and on_segment(p2, q1, q2)):
        return True
    else:
        return False
    
def overlay_mask(frame, mask):
    """
    Overlay a segmentation mask on the input frame.

    Args:
        frame (numpy.ndarray): Input image frame (BGR format).
        mask (numpy.ndarray): Segmentation mask image (single channel).

    Returns:
        numpy.ndarray: Image frame with the mask overlay.
    """
    # Ensure mask is a numpy array
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)

    # Resize the mask to match the dimensions of the frame
    resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # Ensure mask is in binary format (0 or 255)
    mask_binary = (resized_mask > 0).astype(np.uint8) * 255

    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask_binary)

    return masked_frame
    
def save_segmentation_result(frame, mask, bounding_box, filename):
    """
    Apply a resized segmentation mask to the input frame, extract the specified
    region defined by the bounding box, and save the segmented result to a file.

    Args:
        frame (numpy.ndarray): Input image frame (BGR format).
        mask (numpy.ndarray): Segmentation mask image (single channel).
        bounding_box (tuple): Bounding box coordinates (x1, y1, x2, y2).
        filename (str): Output filename for the saved image.
    """
    # Overlay the mask on the frame
    masked_frame = overlay_mask(frame, mask)

    # Extract the segmented region defined by the bounding box
    x1, y1, x2, y2 = bounding_box
    segmented_region = masked_frame[y1:y2, x1:x2]

    # Save the segmented region to the specified filename
    cv2.imwrite(filename, segmented_region)

# Open video capture
video_capture = cv2.VideoCapture('cars_cam_2.mp4')

# Define video output settings
output_file = 'output_video.mp4'
codec = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 codec
fps = int(video_capture.get(cv2.CAP_PROP_FPS))  # Match input video FPS
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

# Initialize VideoWriter object
video_writer = cv2.VideoWriter(output_file, codec, fps, frame_size, isColor=True)

while True:
    ret, frame = video_capture.read()  # Read frame from video capture
    if not ret:
        break  # Break loop if no frame retrieved

    # Perform object detection using YOLO model
    results = model.track(frame, persist=True)

    for result in results:
        boxes = result.boxes
        for i in range(len(boxes.xyxy)):
            box = boxes.xyxy[i]
            object_id = int(boxes.id[i])
            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, box[:4])

            # Check if the object has crossed any path and not already counted
            if (line_intersection((x1, y1), (x2, y2), path1_line_start, path1_line_end) or
                line_intersection((x1, y1), (x2, y2), path2_line_start, path2_line_end) or
                line_intersection((x1, y1), (x2, y2), path3_line_start, path3_line_end)) and object_id not in crossed_objects:
                
                # Update crossed objects set
                crossed_objects.add(object_id)

                # Check which path the object has crossed and increment count accordingly
                if line_intersection((x1, y1), (x2, y2), path1_line_start, path1_line_end):
                    path1_count += 1
                if line_intersection((x1, y1), (x2, y2), path2_line_start, path2_line_end):
                    path2_count += 1
                if line_intersection((x1, y1), (x2, y2), path3_line_start, path3_line_end):
                    path3_count += 1

                bounding_box = (x1, y1, x2, y2)

                # Perform segmentation to get the mask
                seg_results = seg_model(frame, conf=conf)
                # Access the segmentation mask from the dictionary
                for seg_mask in seg_results:
                    if seg_mask.masks is not None:
                        # Retrieve the segmentation mask data
                        mask = seg_mask.masks.data[0]  # Assuming a single mask result

                        # Define the filename for saving the masked crop
                        filename = f'Mask_{object_id}.jpg'

                        # Save the segmentation mask applied to the cropped region
                        save_segmentation_result(frame, mask, bounding_box, filename)
                    else:
                        print("No masks found in segmentation result.")


            # Draw rectangle around the object (orange color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 127, 255), 2)

            # Display object ID, class name, and confidence as labels on the rectangle
            label = f"ID:{object_id} {get_class_name(cls)} {conf:.2f}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 127, 255), -1)  # Orange background
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text

    # Draw paths and labels on the frame
    draw_paths_and_labels(frame)

    # Display counts on the frame
    cv2.putText(frame, f'Car Count Path 1: {path1_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Car Count Path 2: {path2_count}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Car Count Path 3: {path3_count}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Write processed frame to output video
    video_writer.write(frame)

    # Display the processed frame
    cv2.imshow('Traffic Monitoring', frame)

    # Check for 'q' key press to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print crossed objects
print("Crossed Objects", crossed_objects)

# Release video capture and close all windows
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()