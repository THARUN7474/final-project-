# AI-Based Remote Road Mapping using Satellite Imagery
*Updated Presentation Content based on Proposed Enhancements*

---

## Slide 1: Introduction & Problem Statement
The main problem addressed is the severe environmental impact of unregulated road networks in remote and semi-forested tropical regions. These areas suffer from hidden deforestation, illegal land clearing, and uncontrolled road expansion, which have devastating ecological consequences.

Traditional road mapping methods struggle to provide timely and precise data, making it difficult to monitor and manage these vulnerable areas effectively. This project aims to address this issue by leveraging advanced Convolutional Neural Networks (CNNs) to accurately identify and track road expansions, supporting continuous monitoring of human impact in ecologically sensitive regions.

---

## Slide 2: Background & Literature Review
The advancement of automated road detection gained momentum after the 2018 DeepGlobe Road Extraction Challenge. Major developments included modified architectures like D-LinkNet-34 and connectivity-focused models like NodeConnect.

While standard models (like vanilla UNet and ResNet) provide a strong foundation, they often struggle with a critical issue in dense forested areas: **broken road segments** caused by tree canopy occlusion and severe class imbalance. This project builds upon these baseline CNN architectures but introduces advanced topological awareness to ensure continuous, unbroken road mapping.

---

## Slide 3: Project Objectives
1. **Advanced Model Development:** To develop and train a novel **Attention-ResUNet** model that integrates attention gates and residual connections for superior feature extraction on fine road structures.
2. **Robust Loss Optimization:** To implement a custom **Focal Tversky Loss with a Connectivity Penalty** to explicitly address class imbalance and penalize broken road predictions.
3. **Cross-Domain Evaluation:** To compare baseline models (UNet, ResNet-34) with the proposed model across multiple datasets (DeepGlobe and a Custom Pantropical Dataset) using both pixel-wise metrics (IoU, Dice) and topological connectivity evaluation.
4. **Real-World Deployment:** To develop a fully functional Django-based web interface for real-time satellite imagery inference.

---

## Slide 4: Key Highlights & Innovations
* **Attention-ResUNet Architecture:** Enhances detection of narrow, occluded roads by suppressing irrelevant background noise.
* **Connectivity-Aware Loss:** Replaces standard BCE/Soft Dice with Focal Tversky Loss + Connectivity mathematical penalty.
* **Dual-Dataset Generalization:** Rigorously evaluated on both the standard DeepGlobe dataset and a highly challenging Custom Pantropical Dataset.
* **Optimized TensorFlow Pipeline:** Efficient data handling, dynamic batch sizing, and aggressive data augmentation strategies.
* **End-to-End Application:** User-friendly Django web application for practical deployment by environmental monitoring agencies.

---

## Slide 5: System Architecture & Workflow
1. **Data Collection:** High-resolution satellite imagery from trusted sources, utilizing both the DeepGlobe Challenge dataset and a specialized Custom Pantropical Dataset representing dense forested environments.
2. **Data Preprocessing:** Enhancement via TensorFlow data pipelines, applying normalization, aggressive augmentation (random flips, rotations), and handling severe background-to-road class imbalances.
3. **Model Training:** Developing the baseline UNet/ResNet models alongside the proposed Attention-ResUNet architecture, utilizing early stopping and custom loss functions.
4. **Evaluation:** Assessing models with Topological Connectivity Metrics alongside standard Intersection over Union (IoU) and Soft Dice Coefficient.
5. **Deployment:** Integrating trained model weights into a Django web platform, enabling users to upload imagery and receive instant road masks.

---

## Slide 6: Datasets
**DeepGlobe Road Extraction Challenge Dataset**
* Standard benchmark dataset sourced from Kaggle.
* Images are 1024x1024 pixels, consisting of 6,226 training images, validated and tested on an 80:20 split.
* Paired with high-quality binary masks representing road labels.

**Custom Pantropical Dataset (Proposed Addition)**
* Highly challenging imagery focused entirely on remote, dense forested regions to test cross-domain generalization and ensure the model performs in real-world environmental monitoring scenarios.

---

## Slide 7: Methodology & Models Used
**Baseline Architectures:**
* **U-Net:** A CNN designed with an encoder-decoder structure, effective for baseline pixel-wise semantic segmentation.
* **ResNet-34:** Deep architecture utilizing residual connections to allow deeper network training without vanishing gradients.

**Proposed Architecture (Attention-ResUNet):**
* Combines the deep feature extraction of ResNet with the spatial recovery of U-Net, further enhanced by **Attention Gates**. These gates allow the model to specifically "focus" on thin, faint road pixels while ignoring canopy cover and terrain noise.

---

## Slide 8: Evaluation Metrics & Comparative Analysis
The comparative performance of the baseline and proposed models was assessed across training and validation splits. 

* **Standard Metrics:** Evaluated using Intersection over Union (IoU), Dice Coefficient (F1-Score), Precision, Recall, and Loss.
* **Topological Metrics (New Contribution):** Specialized metric to evaluate road continuity and connectivity, proving that the proposed Attention-ResUNet with Focal Tversky Loss significantly reduces fragmentation in mapped roads compared to the baselines.
* *Bar charts in the presentation will clearly illustrate the Attention-ResUNet outperforming baselines across both DeepGlobe and the custom dataset.*

---

## Slide 9: User Interface & Deployment (Django Web App)
To transition this research into a practical tool, a web application was developed.
* **Authentication:** Secure Registration and Login pages requiring Name, Email, Password, and Address.
* **Interactive Dashboard:** Once authenticated, users can upload satellite imagery directly into the portal.
* **Real-time Inference:** The backend immediately processes the image using the highly optimized Attention-ResUNet weights, rendering the predicted road network overlay on the screen.

---

## Slide 10: Conclusion
The proposed AI-based remote road mapping system leverages an advanced **Attention-ResUNet** coupled with a connectivity-aware loss function to accurately extract road topologies from complex satellite imagery. By successfully testing on multiple geographic domains (DeepGlobe + Custom Dataset), the system demonstrates significant improvements in precision, generalization, and continuity over traditional CNN baselines. 

This approach effectively addresses the challenges of unregulated pantropical road development, offering a scalable, reliable web-based solution for sustainable infrastructure planning and environmental preservation.

---

## Slide 11: Future Scope
1. **Unsupervised Learning:** Utilizing self-supervised or unsupervised AI to detect roads without manually labeled masks, making the base model infinitely adaptable to uncharted environments.
2. **Multi-Temporal Analysis:** Implementing time-series remote sensing to automatically track and alert authorities to the *speed* of road expansions over different seasons or years.
3. **Cloud-Based Processing System:** Migrating the Django application to a distributed cloud computing architecture (AWS/GCP) to enable extremely large-scale, automated raster analysis over millions of square kilometers.
