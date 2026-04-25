# Viva & Presentation Preparation Guide: DeepGlobe Road Extraction Project

This document contains a comprehensive list of predicted questions and detailed answers for your final year project presentation and viva. The questions are categorized into logical sections covering dataset, architecture, mathematics, web app, team roles, and future work.

---

## 1. Project Overview & Proposed System

**Q1: What is the main problem your project addresses?**
**Answer:** Our project addresses the lack of accurate, automated road mapping in remote, semi-forested tropical regions. Unregulated road expansion in these areas causes severe environmental degradation (deforestation, illegal mining). Traditional mapping is manual and slow, so we leverage AI to automate this process using satellite imagery.

**Q2: What is your proposed system?**
**Answer:** We propose an AI-based web application that utilizes Deep Learning, specifically Convolutional Neural Networks (CNNs). We trained UNet, ResNet-34, and a proposed enhanced model (ResNet-34+ with residual connections) on high-resolution satellite imagery (DeepGlobe Dataset). We then deployed the best-performing models into a Django-based web application where users can upload satellite images and instantly extract road networks.

**Q3: Why did you choose this approach over existing ones (like Facebook’s D-LinkNet)?**
**Answer:** Standard approaches often train heavily on urban or highly developed regions. Their performance drops in semi-forested or remote areas where roads are faint, narrow, or covered by canopies. Our proposed architecture focuses on these faint topographical features and introduces targeted data preprocessing (like augmentation) to enhance extraction in these difficult terrains.

---

## 2. Dataset & Preprocessing

**Q4: What dataset did you use and what does it look like?**
**Answer:** We used the **DeepGlobe Road Extraction Challenge Dataset**, acquired from Kaggle.
*   **Contents:** It contains pairs of high-resolution true-color satellite images and their corresponding binary mask labels (where white pixels represent the road and black pixels represent the background).
*   **Size:** 6,226 training images, 1,243 validation images, and 1,101 test images. The images are originally 1024x1024 pixels.

**Q5: Why and how did you split the dataset (80:20)?**
**Answer:** We used an **80:20 split** (80% training, 20% validation/testing). 
*   **How:** By randomizing the selection of image tiles to increase diversity and prevent bias.
*   **Why:** The 80% is used by the model to learn hidden patterns and features (the "study" phase). The remaining 20% is unseen evaluation data to test how well the model generalizes (the "exam" phase). If we didn't split it, the model would simply memorize the images (overfitting) and perform poorly on totally new imagery.

**Q6: What preprocessing techniques did you apply to the dataset and images?**
**Answer:**
1.  **Resizing:** We resized images down to 256x256 pixels to make them computationally feasible for training in batches without exceeding GPU limits.
2.  **Normalization:** Pixel values (0 to 255) were divided by 255.0 so they fit into a scale of `[0.0, 1.0]`. Neural networks converge much faster and behave more stably with normalized inputs.
3.  **Data Augmentation:** We applied random horizontal and vertical flips. This artificially increases the dataset variability, ensuring the model doesn't memorize specific road orientations.

---

## 3. Basic ML, Math, Loss Functions & Metrics

**Q7: In Deep Learning, what is an "Epoch" and why did you use 100/1000 epochs?**
**Answer:** An epoch is one complete forward and backward mathematical pass of the *entire* training dataset through the neural network. 
*   We let the model run up to 1000 epochs initially but used a technique called **Early Stopping** (with a patience of 10).
*   **Why:** We let the model train as long as it is learning. If the Validation Loss stops dropping (improving) for 10 consecutive epochs, training is halted automatically. This guarantees we capture the best possible model weights exactly before it starts to overfit.

**Q8: What mathematical loss function did you use? Why?**
**Answer:** We primarily used **Cross-Entropy Loss (Log Loss)** and **Soft Dice Loss**.
*   **Cross-Entropy Math:** $Loss(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} (y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i))$
*   *Why:* It continuously evaluates the probability of a pixel belonging to the 'Road' class (0 or 1). It heavily mathematically penalizes the model when it is highly confident but incorrect.

**Q9: Why use metrics like "F1 Score" and "mIoU" instead of simple "Accuracy"?**
**Answer:** Because of a problem called **Class Imbalance**. In a typical satellite image, over 95% of the pixels might be forest/background, and only <5% are actual roads. If a "dumb" model simply predicts "Not Road" for every single pixel, it will still score 95% Accuracy, yet it failed its task.
*   **mIoU (Mean Intersection over Union):** Calculated as `(Area of Overlap) / (Area of Union)`. This measures strict spatial alignment. If the model is slightly off the road outline, mIoU drops significantly.
*   **F1 Score:** Formula is $2 \times \frac{Precision \times Recall}{Precision + Recall}$. It creates a harmony between successfully finding actual roads (Recall) and not predicting fake roads (Precision).

---

## 4. Models and Architecture (CNNs)

**Q10: What is a CNN and how does it apply here?**
**Answer:** Convolutional Neural Networks (CNNs) are specialized deep learning models for grid-like data (images). They use mathematical convolution operations (filters applying matrix multiplication) moving across an image to automatically learn hierarchical spatial features like edges, dirt textures, and lane lines, without manually coding those rules.

**Q11: Explain your models: UNet vs ResNet-34 vs ResNet-34+.**
**Answer:**
*   **UNet:** Designed specifically for biomedical and topographical semantic segmentation. It is shaped like a 'U' (Encoder-Decoder) and uses "skip connections" to map low-level detail features (from the input) directly over to high-level semantic features (at the output) to draw precise pixel boundaries.
*   **ResNet-34 (Residual Network):** A very deep network that uses "Residual blocks" (skip connections over layers) to bypass calculations. This solves the famous "vanishing gradient problem," where very deep neural networks forget what they learn in early layers.
*   **Proposed Enhanced Model (ResNet-34+):** My proposed system where we integrated the residual encoding strengths of ResNet with the dense up-sampling semantic structure of UNet. This hybrid approach passes dense spatial information across the pipeline specifically to capture broken, discontinuous, or faint jungle roads better than standard models.

---

## 5. The Web Application

**Q12: Why did you build a web application for this? What is its use case?**
**Answer:** An academic model saved as an `.h5` file on a laptop is not accessible to true end-users (like urban planners, government officials, or conservationists). The web app (built in Django) provides a robust, user-friendly interface where a non-technical user can just upload an image, select the model they want, and instantly receive mapped road networks on their screen.

**Q13: How does the data logic flow in your Django web application?**
**Answer:** 
1.  **Input:** User uploads a satellite image on the frontend dashboard. The file goes to the Django backend (`media/uploadedImages/`).
2.  **Model Loading:** The specific view function loads our pre-trained `.h5` weights using TensorFlow Keras.
3.  **Preprocessing:** OpenCV is used in Python to convert the image from BGR to RGB, resize it exactly to 256x256, and normalize pixel values by dividing by 255.
4.  **Inference:** Using NumPy, we expand the image dimension to create a batch of 1 (`(1, 256, 256, 3)`). It is passed through `model.predict()`.
5.  **Output:** The predicted probability array is drawn as a grayscale mask image via Matplotlib, encoded directly into a Base64 string, and rendered side-by-side with the original image on the HTML template.

---

## 6. Code, Problems Faced & Learnings

**Q14: How did you code it and run it?**
**Answer:** 
1.  **Training:** We used Jupyter Notebooks with high-end GPUs for the mathematically heavy training phase of `UNet` and `ResNet`. The finalized weights were exported as `.h5` files.
2.  **Deployment:** Using Visual Studio Code, I developed the Django backend framework. The `.h5` models are placed inside a backend folder (`Deep Globe/`), and the web server runs on local Python, essentially acting as our machine learning inference engine.

**Q15: What were the major problems you faced?**
**Answer:**
1.  **Memory Constraints (OOM Errors):** The satellite dataset had thousands of high-resolution images. Loading them all into RAM crashed the system. I solved this by utilizing TensorFlow Data Pipelines (`tf.data.Dataset`) to stream images in optimized memory batches.
2.  **Capturing Faint Roads:** Roads were frequently vanishing or broken during training. We addressed this by switching implementations, applying data augmentations, and ultimately bringing in the ResNet-34+ architecture to strengthen spatial feature capturing.
3.  **Integrating ML with a Web UI:** Managing Python environment conflicts (TensorFlow vs Django versions) and specifically learning how to return an image from memory (Base64) without constantly writing new image files to the server's hard drive.

**Q16: What were your major learnings doing this project?**
**Answer:** Translating abstract math and theoretical research papers into a fully functional, tangible application. I mastered neural architecture concepts, learned how to manipulate multi-dimensional tensors, and learned full-stack development by creating a fluid pipeline between deep learning models and a Django frontend.

---

## 7. Team Contributions

**Q17: There are 3 members in your team. How was the work divided among you?**
**Answer:**
We divided the responsibilities logically to ensure we extensively tested all parameters.
*   **Member 1:** Was tasked with running, training, and validating the **UNet model**. They monitored how epochs affected UNet's specific architectural loss and extracted evaluation metrics.
*   **Member 2:** Was responsible for compiling, running, and validating the standard **ResNet-34 model**, comparing its efficiency against UNet on the validation images.
*   **Me (Member 3 - Team Lead & Architect):** 
    *   **Architecture & Coding:** I developed the overarching codebase structure, coded the proposed hybrid model (ResNet-34+), and led the full full-stack web integration into Django. 
    *   **Guidance & Oversight:** I explained the dataset logic to the team, demonstrated how the data pipeline and OpenCV worked, and set up the training files so they fully understood what data they were inputting, how to interpret their output `.h5` files, and where their results were being stored.

---

## 8. Future Scope & Generic Questions

**Q18: What is the future scope or "next step" of this project?**
**Answer:**
1.  **Real-time Satellite APIs:** Instead of static uploads, we want to integrate live coordinates with services like Google Earth Engine or Mapbox APIs to actively monitor regions.
2.  **Adding Object Detection:** Extending the CNN logic to not just trace road segments, but use bounding boxes (like YOLO) to identify vehicles, illegal deforestations, or buildings located on those roads.
3.  **Global Scale Generalization:** Our model focuses heavily on semi-forested areas. Future work entails training a gigantic ensemble model against various global biomes (snowy mountains, deserts, urban jungles) to create a universal mapping tool.

**Q19: If someone uploads a normal photo (not a satellite image), what happens to your model?**
**Answer:** The model would attempt to apply its convolutional filters to the image, likely outputting garbage results or tiny random white pixels. The model is specifically trained on top-down topological structures, colors, and textures belonging specifically to satellite photography. It has no conceptual understanding of a normal horizontal photo. 
