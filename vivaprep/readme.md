# DeepGlobe Road Extraction Web Application Overview

This document provides a comprehensive overview of your codebase, explaining its structure, the technologies used, the Machine Learning (CNN) implementation, and step-by-step instructions on how to set it up and run it.

## 1. Project Overview
This project is a **Django-based web application** that allows users to upload satellite imagery and perform **Road Extraction (Semantic Segmentation)** using Convolutional Neural Networks (CNNs). It includes user authentication (with OTP verification), profile management, a dashboard, and a deep learning inference module that loads pre-trained CNN models (U-Net and ResNet) to predict road masks on uploaded images.

The models are likely trained on the **DeepGlobe Road Extraction Dataset**, a popular dataset in remote sensing and computer vision.

---

## 2. Folder Structure Explanation

Here is a breakdown of what each folder and significant file in your repository `c:\Users\Tharun\dgb-C\` does:

*   **`baseapp/`**: A Django application that handles the core unauthenticated functionalities. It includes:
    *   Index, About, and Contact pages.
    *   User Registration and Login logic.
    *   OTP generation and verification (via email and SMS using `smslogin.co`).
    *   Admin login routing.
*   **`userapp/`**: A Django application for authenticated users. It manages:
    *   User Dashboard, User Profile, and Feedback system.
    *   **`views.py -> user_detection`**: The core Machine Learning inference logic. It takes image uploads, loads the `.h5` model, processes the image, runs the prediction, and returns the result.
*   **`adminapp/`**: A Django application for the administrator portal (managing users, viewing feedback, etc.).
*   **`deep_globe_project/`**: The main Django project folder containing configurations (`settings.py`, root `urls.py`, etc.).
*   **`Deep Globe/`**: The Machine Learning core. It contains:
    *   `deepglobe-resnet.ipynb` & `deepglobe-unet.ipynb`: The Jupyter notebooks used to train the implementation of the research papers.
    *   `road_extraction_Resnet (1).h5` & `road_extraction_unet.h5`: The actual trained model weights that the web application uses to make predictions.
*   **`media/` & `static/`**:
    *   `media/`: Stores user uploads (like profile pictures and `uploadedImages/` for detection).
    *   `static/`: Stores CSS, JavaScript, and static images for your frontend design.
*   **`dataset_images/`**: Contains raw or sample images from the dataset used for testing or training.
*   **`requirements.txt`**: A list of all Python libraries needed to run the project.
*   **`db.sqlite3` & `deep_globe_db.sql`**: The local database file and an SQL dump file for the database schema.
*   **`manage.py`**: The standard Django command-line utility used to start the server, make migrations, etc.

---

## 3. Technologies & Languages Involved

### Languages
*   **Python 3**: The core language for both the web backend and the Machine Learning models.
*   **HTML, CSS, JavaScript**: The languages used for the frontend HTML templates.
*   **SQL (SQLite/MySQL)**: Database querying language.

### Frameworks & Libraries
*   **Django (v5.1.6)**: The backend web framework used to build applications, handle routing, user sessions, and abstract the database.
*   **TensorFlow (v2.18.0) & Keras (v3.8.0)**: The Deep Learning frameworks used to build, train, and run the CNN models.
*   **OpenCV (`opencv-python`)**: Used for image processing (reading, resizing, color conversion).
*   **Matplotlib**: Used to visualize model predictions and convert them into base64 images so they can be rendered in the browser.
*   **NumPy**: Used for matrix operations and manipulating image arrays.

---

## 4. The Machine Learning (CNN) Implementation

The project implements semantic segmentation to extract roads from satellite images. Here is how the pipeline works inside `userapp/views.py -> user_detection`:

1.  **Selection**: The user selects either the **U-Net** or **ResNet** model from the frontend form.
2.  **Upload**: The script saves the uploaded image to the `media/uploadedImages/` folder.
3.  **Model Loading**: It uses `tf.keras.models.load_model()` to load either `road_extraction_unet.h5` or `road_extraction_Resnet (1).h5` from the `Deep Globe` folder.
4.  **Preprocessing (`read_image` function)**:
    *   The image is loaded using OpenCV.
    *   Color format is converted from BGR (OpenCV default) to RGB.
    *   It's explicitly resized to `(256, 256)` pixels to match the input shape requirement of the CNN.
    *   The pixel values are normalized to a scale of `[0.0, 1.0]` by dividing by 255.0.
5.  **Inference**:
    *   The image dimension is expanded using `np.expand_dims()` (e.g., from `(256,256,3)` to `(1,256,256,3)`) to create a "batch" of 1 image.
    *   `model.predict()` is called to generate the road mask.
6.  **Post-processing**: The generated mask prediction is rendered using Matplotlib as a grayscale image (`cmap='gray'`), converted to a Base64 string, and passed back directly to the web template to be displayed next to the original image.

### About the Models
*   **U-Net**: A classic CNN architecture designed for biomedical image segmentation, highly effective at finding precise localized boundaries (roads).
*   **ResNet**: Utilizes residual connections to build deeper networks without suffering from vanishing gradients, feature-extracting standard images very effectively.

---

## 5. How to Setup and Run the Project

### Prerequisites
*   Ensure you have **Python** (preferably Python 3.9 to 3.11 for TensorFlow compatibility) installed.
*   Ensure you have **pip** installer available.

### Step-by-Step Setup

1.  **Activate Virtual Environment (Optional but Recommended)**
    If `env` is your virtual environment, activate it:
    *   *Windows*: `.\env\Scripts\activate`

2.  **Install Requirements**
    Run the following command to download all necessary libraries in the root directory:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables for Email**
    The app sends OTP emails. In `baseapp/views.py`, it expects system environment variables for email configurations. You will need to export them in your terminal before running:
    *   *Windows Command Prompt*:
        ```cmd
        set EMAIL_HOST_USER=your_email@gmail.com
        set EMAIL_HOST_PASSWORD=your_email_app_password
        ```
    *(Note: For testing, if you don't care about emails being actually sent, you can temporarily comment out the `send_mail` functions in `user_register` and `user_login` views or configure a local dummy SMTP server in Django).*

4.  **Database Migrations**
    Sync your models with the database (run this to be safe, although `db.sqlite3` is already present):
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

5.  **Run the Server**
    Start the local Django development server:
    ```bash
    python manage.py runserver
    ```

6.  **Access the Application**
    Open your web browser and navigate to:
    `http://127.0.0.1:8000` or `http://localhost:8000`

### Testing the ML Pipeline
Once running, register an account, verify the OTP, login, and navigate to the "Detection" or feature section of the dashboard. Upload an image from the `dataset_images` folder, select "U-Net" or "ResNet", and submit. You will see the base image alongside the predicted extracted road mask!
