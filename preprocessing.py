import os
import cv2

minValue = 70


def preprocess_image(image_path):
    frame = cv2.imread(image_path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return res


def preprocess_images_in_folder(folder_path):
    for class_label in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, class_label)

        for filename in os.listdir(class_folder):
            if filename.endswith(".jpg"):
                image_path = os.path.join(class_folder, filename)
                print(f"Processing image: {image_path}")
                processed_image = preprocess_image(image_path)

                # Save the processed image (you can modify this based on your needs)
                processed_folder = os.path.join("processed_data", class_label)
                os.makedirs(processed_folder, exist_ok=True)

                processed_image_path = os.path.join(processed_folder, filename)
                cv2.imwrite(processed_image_path, processed_image)
                print(f"Processed image saved to: {processed_image_path}")


# Apply preprocessing to training data
preprocess_images_in_folder("dataSet/trainingData")

# Apply preprocessing to testing data
preprocess_images_in_folder("dataSet/testingData")
