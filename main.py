import cv2
import ultralytics
import numpy as np
import matplotlib.pyplot as plt

def load_model(model_path="best.pt"):
    """
    Load the YOLO model from the specified path.
    """
    try:
        model = ultralytics.YOLO(model_path)
        print(f"Model '{model_path}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def read_image(image_path):
    """
    Read an image from the filesystem.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image from '{image_path}'. Please check the path.")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
        print(f"Image '{image_path}' loaded successfully.")
        return image
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def perform_detection(model, image):
    """
    Perform object detection on the image using the loaded model.
    """
    try:
        results = model(image)
        return results
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

def parse_results(results, detect_all=False):
    """
    Parse the detection results and extract object classes.
    If detect_all is True, return all unique detected objects.
    Otherwise, return the first detected object.
    """
    if not results:
        print("No results to parse.")
        return {}

    json_response = {"objects": []}

    for result in results:
        # Each 'result' corresponds to an image (since we have one image, it's one result)
        if result.boxes is not None and len(result.boxes) > 0:
            class_indices = result.boxes.cls.cpu().numpy().astype(int)  # Convert to integer class indices
            class_names = [result.names[idx] for idx in class_indices]

            if detect_all:
                unique_objects = list(set(class_names))
                json_response["objects"].extend(unique_objects)
            else:
                # Assuming you want the first detected object
                json_response["objects"].append(class_names[0])

            # Optionally, you can include confidence scores and bounding boxes
            # For simplicity, we're only extracting class names here
        else:
            json_response['error'] = 'No objects detected.'

    return json_response

def visualize_detections(image, results, save_path=None):
    """
    Draw bounding boxes and class labels on the image and display it.
    Optionally, save the visualized image to the specified path.
    """
    try:
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_idx = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_name = result.names.get(cls_idx, "Unknown")

                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                # Put label
                label = f"{class_name} {confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

        # Save the image if save_path is provided
        if save_path:
            # Convert back to BGR for saving with OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image_bgr)
            print(f"Visualized image saved to '{save_path}'.")

    except Exception as e:
        print(f"Error during visualization: {e}")

def main():
    # Paths (modify these paths as needed)
    model_path = "best_1.pt"  # Ensure this path points to your trained YOLO model
    image_path = "images/img6.jpg"  # Replace with your test image path
    save_visualization = "detected_image.jpg"  # Path to save the visualized image

    # Load the model
    model = load_model(model_path)
    if model is None:
        return

    # Read the image
    image = read_image(image_path)
    if image is None:
        return

    # Perform detection
    results = perform_detection(model, image)
    if results is None:
        return

    # Parse results (change detect_all to True to get all unique objects)
    json_response = parse_results(results, detect_all=True)
    print("Detection Results:", json_response)

    # Visualize detections
    visualize_detections(image, results, save_path=save_visualization)

if __name__ == "__main__":
    main()
