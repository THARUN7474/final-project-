&lt;img&gt;remote sensing&lt;/img&gt;
&lt;img&gt;MDPI&lt;/img&gt;

Technical Note

# Mapping Remote Roads Using Artificial Intelligence and Satellite Imagery

Sean Sloan <sup>1,2,*,†</sup>, Raiyan R. Talkhani <sup>3,†</sup>, Tao Huang <sup>3</sup>, Jayden Engert <sup>1</sup> and William F. Laurance <sup>1</sup>

<sup>1</sup> Centre for Tropical Environmental and Sustainability Science, College of Science and Engineering, James Cook University, Cairns, Queensland 4878, Australia; jayden.engert@my.jcu.edu.au (J.E.); bill.laurance@jcu.edu.au (W.F.L.)
<sup>2</sup> Department of Geography, Vancouver Island University, Nanaimo, BC V9R 5S5, Canada
<sup>3</sup> College of Science and Engineering, James Cook University, Cairns, Queensland 4878, Australia; raiyanriyaztalkhani@my.jcu.edu.au (R.R.T.); tao.huang1@jcu.edu.au (T.H.)
* Correspondence: sean.sloan@viu.ca
† These authors contributed equally to this work.

**Abstract:** Road building has long been under-mapped globally, arguably more than any other human activity threatening environmental integrity. Millions of kilometers of unmapped roads have challenged environmental governance and conservation in remote frontiers. Prior attempts to map roads at large scales have proven inefficient, incomplete, and unamenable to continuous road monitoring. Recent developments in automated road detection using artificial intelligence have been promising but have neglected the relatively irregular, sparse, rustic roadways characteristic of remote semi-natural areas. In response, we tested the accuracy of automated approaches to large-scale road mapping across remote rural and semi-forested areas of equatorial Asia-Pacific. Three machine learning models based on convolutional neural networks (UNet and two ResNet variants) were trained on road data derived from visual interpretations of freely available high-resolution satellite imagery. The models mapped roads with appreciable accuracies, with F1 scores of 72–81% and intersection over union scores of 43–58%. These results, as well as the purposeful simplicity and availability of our input data, support the possibility of concerted program of exhaustive, automated road mapping and monitoring across large, remote, tropical areas threatened by human encroachment.

**Keywords:** convolutional neural networks; roads; remote sensing; road map; tropical forests; artificial intelligence

---

☑ check for updates
**Citation:** Sloan, S.; Talkhani, R.R.; Huang, T.; Engert, J.; Laurance, W.F. Mapping Remote Roads Using Artificial Intelligence and Satellite Imagery. *Remote Sens.* **2024**, *16*, 839. https://doi.org/10.3390/rs16050839

Received: 9 January 2024
Revised: 16 February 2024
Accepted: 24 February 2024
Published: 28 February 2024

Academic Editor: Yang-Won Lee

---

## 1. Introduction

The Earth is experiencing an unprecedented wave of road building, with some 25 million kilometers of new paved roads expected by mid-century, relative to 2010 [1]. Roughly nine-tenths of all road construction is occurring in developing nations [2,3], including many tropical and subtropical regions of exceptional biodiversity [4–6]. By sharply increasing access to formerly remote natural areas, poorly regulated road development triggers dramatic increases in environmental disruption through economic activities such as logging, mining, and land-clearing [3]. Efforts to plan or zone road development have historically been most inadequate in remote rural areas, wilderness frontiers, and partially intervened natural areas (hereafter semi-forested areas) where road development is most haphazard and environmentally destructive [7–9]. Many roads in such regions, both legal and illegal, are unmapped [10,11]. Hence, road-mapping studies in the Brazilian Amazon [10,12–15], Asia-Pacific [11,16,17], and elsewhere [18,19] regularly find 2–13 times more road length than reported in government sources or online road databases. The abundance of such clandestine roadways underscores the degree to which environmental governance and conservation advocacy are challenged by the lack of complete, up-to-date information on road development [20].

&lt;img&gt;CC BY&lt;/img&gt;
**Copyright:** © 2024 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/licenses/by/4.0/).

<footer>Remote Sens. 2024, 16, 839. https://doi.org/10.3390/rs16050839</footer>
<footer>https://www.mdpi.com/journal/remotesensing</footer>

<header>Remote Sens. 2024, 16, 839</header>
&lt;page_number&gt;2 of 18&lt;/page_number&gt;

Road mapping has long been a tedious, painstaking exercise, ultimately limited in its spatial and temporal coverage accordingly. Traditionally, and still today, road mapping entailed the visual interpretation and manual digitization of road features in satellite imagery [5,11,16,21–24]. This approach is exceedingly laborious, limiting its application to select areas of interest and discouraging the monitoring of road development. More recently, ‘crowdsourced’ road data compiled in OpenStreetMap (OSM, https://www.OSM.org) has offered a promising alternative, whereby OSM users collectively digitize or otherwise add road features to the OSM online global database. For environmental science and governance, however, OSM road data have historically presented major limitations, foremost being relatively scant and/or inconsistent coverage of remote semi-forested areas [11], as well as the inability to focus mapping on particular regions of interest. A recent comparison of visually digitized road features against human-curated OSM road data¹ across Indonesia, Malaysia Borneo, and New Guinea [16] found the former to have three times the length, underscoring the extent of omissions in the OSM database.

Hence, there has been a longstanding call for automated approaches to road mapping at large scales as a means of improved environmental monitoring [21,25]. Recent developments in artificial intelligence have responded with road-mapping algorithms applied to satellite imagery [26]. Machine learning (ML) road mapping employing convolutional neural networks (CNNs) [27–29] has proven successful, amongst various other approaches [26]. Experimentation with ML road mapping has however focused largely on urban and suburban settings [30–33] or densely settled rural areas [34,35]. Roads there are relatively uniform and distinctive from those in remote semi-forested tropical regions characterized by irregular, rustic, and/or faint roads of diverse earthen materials and situated variously within forests, disturbed vegetation, and farms.

Developments in automated road detection accelerated following the 2018 DeepGlobe Road Extraction Challenge (http://deepglobe.org/challenge.html, accessed on 1 August 2023), culminating in Facebook developing a modified D-LinkNet-34 ML model to map roads globally on the basis of high-resolution satellite imagery [36,37]. Though general accuracies of this model are appreciable [36], the fidelity of its output road data is uncertain for remote semi-forested tropical areas specifically, given the exclusion of such areas from the model’s training dataset [36]. Excitingly, Botelho et al. [13] recently used a UNet ML model to map roads across remote semi-forested areas of Brazil on the basis of Sentinel-2 satellite imagery. Accuracies were respectable but depressed by omission errors inherent to their road-reference or ‘testing’ dataset, reflecting its basis in moderate-resolution Landsat imagery [13]. In remote semi-forested tropical contexts specifically, the ultimate accuracy of UNet and similar ML road-detection models therefore remains relatively uncertain.

In this context, we revisit the UNet model of Botelho et al. [13] as well as two alternative ML road-mapping models to clarify their accuracy in remote semi-forested areas, here in equatorial Asia-Pacific. Our study complements Botelho et al. [13] in three key respects. First, we consider an exhaustive road-reference dataset based on high-resolution imagery to ensure confident measures of map accuracy. Second, we include complementary, relatively conservative measures of map accuracy. Third, we base our models on simple ‘screenshots’ of high-resolution satellite imagery freely accessible via Google Earth or similar online geospatial platforms. This experimentational use of such imagery explicitly contemplates the possibility of an open-access scientific program whereby the scientific community may avail of an online ML model coupled with such imagery to map and monitor roads in any region of interest, cf. [38].

## 2. Materials and Methods

### 2.1. Overview

Across equatorial Asia-Pacific, we trained three ML models to automatically map road features on the basis of freely available ‘screenshots’ of high-resolution satellite imagery. Here, we describe these models and report their accuracies. Of our three models, the UNet model is analogous to that employed by Botelho et al. [13] for Brazil. The two other models,

Remote Sens. 2024, 16, 839
&lt;page_number&gt;3 of 18&lt;/page_number&gt;

based on the ResNet-34 architecture [39], offer enhancements to the UNet model while preserving computational efficiency, a factor of likely importance for any potential large-scale scientific road-mapping initiative. Model training and testing were based on a visually interpreted reference dataset of road features across equatorial Asia-Pacific. The accuracy of each model was evaluated using three metrics providing complementary insight into model performance. Thus, we describe a baseline of model performance given standard ML models applied to remote semi-forested tropical areas.

### 2.2. Satellite Imagery and Road Reference Data

#### 2.2.1. Study Area

This study covers rural, generally remote, and often forested areas of equatorial Asia-Pacific (Papua New Guinea, Indonesia, and Malaysia) (Figure 1). This study area was defined on the basis of recent research describing spontaneous and planned road developments in the region, typically in areas characterized by extensive intact or fragmented forest cover [20,24,40–44].

&lt;img&gt;
Observation areas
Intact forest
Degraded/secondary forest
Natural non-forest
Plantations
Cropping/Other deforested
Urban
Water
&lt;/img&gt;

**Figure 1.** Study area encompassing the 200 sampled satellite images. Notes: Land cover data are after [16].

#### 2.2.2. Satellite Imagery

We obtained 200 satellite images for model training, validation, and testing. Images were ‘screenshots’ (i.e., reduced-resolution copies) of high-resolution true-color satellite imagery (~0.5–1 m pixel resolution) observed using the Elvis Elevation and Depth spatial data portal (https://elevation.fsdf.org.au/, accessed on September 2022), which here is functionally equivalent to the more familiar Google Earth. Each of our 200 images were initially acquired at a resolution of 1920 × 886 pixels. Actual image resolution was coarser than the native high-resolution imagery, at 5 m, but still appreciable (Figures 2 and 3a–c). These images are freely available online [45]. The images generally spanned either forest-agricultural mosaics (Figure 3a) or intact forest landscapes with limited human intervention (Figure 3b,c). The 200 images were in PNG file format and ultimately parsed into their constituent red, blue, and green (RBG) channels for model training and road classification.

Remote Sens. 2024, 16, 839
&lt;page_number&gt;4 of 18&lt;/page_number&gt;

&lt;img&gt;Figure 2. A sampled image at full extent (top) and for a smaller inset area (bottom) featuring clearly discernible land covers and road infrastructure.&lt;/img&gt;

&lt;img&gt;Figure 3. Examples of sampled images (a–c) and corresponding reference road data (d–f).&lt;/img&gt;

Remote Sens. 2024, 16, 839
&lt;page_number&gt;5 of 18&lt;/page_number&gt;

### 2.2.3. Road Reference Data

Road features were visually interpreted and manually digitized to create a reference dataset by which to train, validate, and test the road-mapping models (Figure 3d–f). The reference dataset of road features was digitized in each of the 200 true-color images using the ‘pen tool’ in Adobe Photoshop. The pen’s ‘width’ was held constant over varying scales of observation (i.e., image ‘zoom’) during digitization. Consequently, at relatively small scales at least, digitized road features likely incorporate vegetation immediately bordering roads. The resultant binary (Road vs. Not Road) reference images were saved as PNG images with the same image dimensions as the original 200 images.

The 200 satellite images (Figure 3a–c) and corresponding road-reference images (Figure 3d–f) were then subdivided into thousands of smaller image ‘tiles’ of 256 × 256 pixels each. The resultant number of input tiles was subsequently increased using data augmentation procedures [46] meant to enhance the informational basis of neural network training, much as for supervised learning generally. Data argumentation [46] entails a variety of operations applied to image data to produce new, complementary image data, e.g., image rotation, color adjustment. In this work, image rotation was employed, resulting in a total of 8904 image tiles. Of these 8904 image tiles, we randomly selected 80% for model training (during which a model ‘learns’ to recognize road features in the input imagery), 10% for model validation (during which model parameters are iteratively refined), and 10% for final model testing (during which the final accuracy of the output road map is assessed). By randomizing the selection of image tiles, we increased the diversity of data used during training, validation, and testing, a factor found to enhance model accuracy more than the nominal quantity of input data [36]. Sloan et al. [45] provide these 8904 image tiles as true-color images and corresponding road-reference images, allowing for further model development by others.

### 2.3. Machine Learning Models for Road Mapping

#### 2.3.1. UNet Model

Our UNet model derives from the architecture introduced by Ronneberger et al. [47] and substantially resembles the framework used by Botelho et al. [13] to delineate roads in the Brazilian Amazon. Our model embodies two principal stages: the encoding phase, synonymous with down-sampling; and the subsequent decoder phase, colloquially referred to as up-sampling stages (Figure 4). In the encoding phase, a three-channel RGB image is input into the model for encoding. This phase comprises four integral modules, each encompassing two layers, characterized by 3 × 3 convolutional operations devoid of padding. Each convolutional layer is immediately succeeded by a rectified linear activation function (ReLU). Subsequently, a 2 × 2 max-pooling layer, configured with a stride of 2, is applied to the module’s output. The culmination of this phase yields an encoded image referred to as feature channels, progressively doubling subsequent to each module. The post-module feature map tally reads as follows: 64, 128, 512, and 1024 (Figure 4).

The decoding phase of the UNet architecture similarly consists of four discrete modules, each housing two 3 × 3 convolutional layers preceded by ReLU activation. Distinct from the encoding stage, the decoding phase incorporates distinct operations before and after each module. In particular, the input of each decoding module is concatenated with the output stemming from the subsequent encoding module. This intermodular concatenation integrates the input module’s 512 feature channels with the output of the corresponding encoding module, yielding an identical count of 512 layers. A pivotal operation in the decoding phase is the application of a 2 × 2 transposed convolution operation, synonymous with a deconvolutional or up-convolutional layer. This operation reduces the feature map quantity by half while concurrently doubling the dimensional extent of individual feature maps. Our UNet model attempts to skirt issues of small dataset and low accuracy more common to fully convolutional network models [48] by adding skip connections between the down-sampling and the up-sampling phases (Figure 4). The skip connections

Remote Sens. 2024, 16, 839
&lt;page_number&gt;6 of 18&lt;/page_number&gt;

transferred information from the feature extraction layers to the up-sampling layers by concatenating data in the encoding phase to data in the decoding phase at the same level.

&lt;img&gt;UNet model architecture diagram&lt;/img&gt;

Figure 4. UNet model architecture as adopted by the present study.

In the ultimate stride of the architectural flow, a 1 × 1 convolutional operation is executed on the concluding layer (Figure 4). This operation elicits a reduction in the number of feature maps to align with the cardinality of the objects under classification, an unequivocal 1 in the context of this study, given the binary classification of Road vs. Not Road. Scripts for this UNet model and the other models discussed below were composed in the Python programming language using TensorFlow libraries.

### 2.3.2. ResNet-34 Model

The ResNet-34 model architecture here similarly has two main phases: encoding and decoding. ResNet-34’s encoding stage consists of 16 modules (pink boxes in Figure 5), each having 2 convolutional layers with a 3 × 3 kernel and ReLU activation function. Each module’s output was combined with its input through residual connections (aka ‘skip connections’). A max pooling operation with a stride of 1 was conducted after each module’s convolutional computation, before data propagation to the next module. Modules without residual connections are where average pooling operations occurred. An important aspect of the encoding phase is the strategic use of max pooling operations with stride 2 (Figure 5), which reduced the dimensionality of feature maps by half and doubled their number. The resulting feature maps were enumerated as 64, 128, 256, and 512, reflecting their cardinality as they evolved throughout the encoding phase.

Remote Sens. 2024, 16, 839
&lt;page_number&gt;7 of 18&lt;/page_number&gt;

&lt;img&gt;ResNet-34 model architecture diagram&lt;/img&gt;

Figure 5. ResNet-34 model architecture as adopted by the present study.

The ResNet-34 architecture, initially designed for image classification, was modified here to enable semantic segmentation. In the original ResNet-34 architecture, fully connected layers were used in the output, which cannot be used here for pixel-wise classification of the Road vs. Not Road classes. Instead, fully connected layers our ResNet-34 model were replaced with three consecutive up-sampling layers with a stride of 2, i.e., deconvolutions, to resize the output to the original size of the image (blue boxes in Figure 5). Each of these up-sampling layers were smoothly integrated with a 2 × 2 transpose convolution operation. This process was used to simplify the output feature map while maintaining its original dimensions. The model’s focus is on binary classification, again Road vs. Not Road, evaluated at each pixel.

The ResNet-34 architecture was preferred over more complex, ‘deeper’ variations, such as the ResNet-110 architecture with 110 layers, because of its greater balance of computational efficiency and model accuracy. Efficiency is potentially an important factor for any scientific open-access and/or online ML road-detection initiative realized at regional to continental scales. Our findings, based on ResNet-34 as well as UNet, therefore represent a baseline against which more complex models prioritizing accuracy over efficiency may be considered.

2.3.3. Resnet-34 Model with Added Residual Connections (ResNet-34+)

The ResNet-34+ model architecture is based on the ResUNet-a architecture described by Diakogiannis et al. [30]. Its architecture’s encoding phase here was taken from the ResNet-34 model (Figure 5) and similarly consists of 16 modules (pink boxes in Figure 6). Relative to the ResNet-34 architecture (Figure 5), residual connections were added between each of the max pooling layers and the up-sampling layers to preserve the data between the encoding and decoding layers to produce a more accurate segmentation map (Figure 6). The output of the residual connections was added to each of the up-sampling layers, unlike the concatenation method used in the UNet architecture. Specifically, connections were made between the 1st max pooling operation and the 3rd up-sampling layer, the 2nd max pooling

Remote Sens. 2024, 16, 839
&lt;page_number&gt;8 of 18&lt;/page_number&gt;

operation and the 2nd up-sampling layer, and the 3rd max-pooling operation and the 1st up-sapling layer (Figure 6). Layers were joined using the concatenation operation, as for the UNet architecture. Compared to the ResUNet-a architecture [30], ResNet-34+ here featured fewer up-sampling operations in order to preserve the data of the up-sampling stages.

&lt;img&gt;ResNet-34+ model architecture diagram&lt;/img&gt;

Figure 6. ResNet-34+ model architecture as adopted by the present study.

### 2.4. Model Training and Validation

For the training of the UNet, ResNet-34, and ResNet-34+ models, no pretrained model was used so that model performance could be readily compared. Model training was broken down into two stages, the first determining pretrained weights and the second determining final weights. Random numbers were assigned for the initial weight values rather than zeros or any other uniform number.

In the initial stage of training, a model was trained for up to 1000 epochs. Each epoch entailed traversing through the entire dataset for model training, validation, and testing. A call-back function with a patience parameter of 10 epochs monitored the model’s validation loss trajectory. If no progress was observed in validation loss over the last 10 epochs, or if there was an increase in validation loss (indicating model overfit), the models’ weights were deemed optimal and the training was terminated (Figure S1). This call-back featured reduced the time required for model training if optimal values were attained before all 1000 epochs were traversed. The trained weights were saved for future training instances.

Our models’ loss trajectory was given by the cross-entropy loss function [32,33,49] (Equation (1)), also known as log loss. This function summarizes the classification performance of a model whose outputs are probabilities. It increases proportionally to the magnitude of discrepancy between predicted and actual probabilities of class membership, here being *Road* and *Not Road*. This function therefore reflects not only the frequency of misclassification but also the degree to which a model mis-estimates the probability of class membership, with increasingly larger discrepancies being penalized increasingly by its logarithmic function (Equation (1)). In our loss function, $\hat{y}$ refers to the predicted

Remote Sens. 2024, 16, 839
&lt;page_number&gt;9 of 18&lt;/page_number&gt;

probability of class membership (*Road* vs. *Not Road*), *y* refers to the true value of the pixel label, *N* refers to the image-tile batch size, and *i* refers to the index.

$$
\text{Loss}(\hat{y}, y) = -\frac{1}{N} \sum_{i=0}^{N} \log(\hat{y}^i) + (1 - y^i) \log(1 - \hat{y}^i) \quad (1)
$$

The second stage of model training utilized pre-primed models from the first stage, rather than starting from scratch. To avoid overfitting, image tiles from the training folder, being a random set of all image tiles, were randomized with respect to their ordering so that the same batches of image tiles were not used to train the model again. The training epoch count was reduced to 500, the patience value retained as 10, and the training process restarted. The lower number of epochs in the second training stage reflected the expectation that fewer iterations were necessary to reach optimal parameters. For both stages of model training, random numbers were assigned for the initial values of weights, instead of zeros.

### 2.5. Model Testing

Two complementary metrics tested the three models’ final road-mapping accuracies: the F1 score and mean intersection over union.

#### 2.5.1. F1 Score of Model Accuracy

The F1 score (Figure 7) describes a model’s accuracy in classifying the target class (*Road*) while accounting for the inevitably imbalanced nature of our reference data, whereby pixels of the target class (*Road*) occur far less frequently than the background class (*Not Road*). Accounting for such class imbalance prevents any inflation of reported accuracy due to the gross under-prediction of the target class or gross over-prediction of the background class. The F1 score accounts for class imbalanced by incorporating measures of model recall (also known as producer’s accuracy) and model performance (also known as user’s accuracy) (Equation (2)). The F1 score has theoretical minima and maxima of 0 and 1, where 1 indicates the perfect prediction of the known road features in the reference dataset.

$$
\text{F1 score} = 2 \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \quad (2)
$$

&lt;img&gt;Illustration of the F1 score of model accuracy. The diagram shows a known road area and a predicted road area. The area of overlap between the two is labeled "Area of Overlap". The formula for F1 Score is given as (Area of Overlap x 2) / Total Area. The total area is the sum of the known road area and the predicted road area. The area of overlap is also shown as 2x.&lt;/img&gt;

Figure 7. Illustration of the F1 score of model accuracy.

In Equation (2), the precision term describes how frequently a model’s classification of *Road* is, in fact, *Road*. Precision is given by the ratio of the frequency of true positives (TPs) to the combined frequency of true positives and false positives (FPs), i.e., all pixels labelled as roads, correctly or incorrectly (Equation (3)). Conversely, the recall term in Equation (2) describes how frequently a model’s classification of *Road* reflects the known extent of *Road*. Recall is given by the ratio of the frequency of true positives to the combined frequency of true positives and false negatives (FNs), i.e., all pixels that are known to be roads (Equation (4)).

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \quad (3)
$$

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} \quad (4)
$$

<header>Remote Sens. 2024, 16, 839</header>
&lt;page_number&gt;10 of 18&lt;/page_number&gt;

### 2.5.2. Mean Intersection over Union Metric of Model Accuracy

The mean intersection over union (mIoU) metric describes the degree to which image features classified as *Road* spatially overlap actual roads in the reference data but not areas known to be *Not Roads* in the same reference data. Given our *Road* target class, the mIoU metric is given formally as the ratio of, on the one hand, the area of overlap of predicted and known *Road* pixels and, on the other, the combined area of predicted and known *Road* pixels, averaged over all N image tiles (Figure 8, Equation (5)) [29]. This metric is similar to the F1 score in that it accounts for the imbalanced nature of the reference data. The mIoU metric has a theoretical minima and maxima of 0 and 1, where 1 indicates an exact duplication of the road features in the reference dataset. Equation (5) may be simplified as Equation (6).

$$
\text{mIoU} = \frac{1}{N} \sum_{i=0}^{N} \frac{\text{Predicted Road} \cap \text{Known Road}}{\text{Predicted Road} \cup \text{Known Road}} \quad (5)
$$

$$
\text{mIoU} = \frac{1}{N} \sum_{i=0}^{N} \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}} \quad (6)
$$

&lt;img&gt;Illustration of the intersection over union (IoU) metric of model accuracy. The diagram shows the formula IoU = Area of Overlap / Area of Union. It depicts two overlapping rectangles labeled "Known Road" and "Predicted Road". The overlapping region is labeled "Area of Overlap" and the total area of both rectangles combined is labeled "Area of Union".&lt;/img&gt;

**Figure 8.** Illustration of the intersection over union (IoU) metric of model accuracy.

### 3. Results

Figure 9 reports road-mapping accuracy amongst of our three models according to the F1 score and mIoU metrics (Equations (2) and (6)). These metrics' values are middling but indicative of accurate road detection.

&lt;img&gt;Bar chart titled "Accuracy Metric". The y-axis is labeled "Accuracy" and ranges from 0% to 100%. The x-axis has two categories: "F1 Score" and "mIoU". There are three bars for each category, representing UNet (yellow), ResNet-34 (orange), and ResNet-34+ (brown). For F1 Score, UNet is approximately 72%, ResNet-34 is approximately 81%, and ResNet-34+ is approximately 81%. For mIoU, UNet is approximately 65%, ResNet-34 is approximately 70%, and ResNet-34+ is approximately 70%.&lt;/img&gt;

**Figure 9.** Road classification accuracy by model.

The F1 scores ranged from 72% for UNet to 81% for ResNet-34 and ResNet-34+. The lower F1 score for our UNet model was greater than that of the UNet road-mapping model of Botelho et al. [13], at 65–68%, the latter having been similarly developed for remote semi-forested areas in the Brazilian Amazon using 10-meter Sentinel-2 satellite imagery. Our greater UNet accuracy compared to Botelho et al. [13] is probably due mostly to the greater accuracy of our road-reference data and the finer resolution of our satellite data,

Remote Sens. 2024, 16, 839
&lt;page_number&gt;11 of 18&lt;/page_number&gt;

given the comparability between our study and Botelho et al. [13] in terms of UNet model design and study context. The higher F1 scores of 81% for our two ResNet models are consistent with the F1 scores of a diverse and often relatively sophisticated range of ML road-detection models reviewed by Abdallahi et al. [26]. The reviewed models vary by deep learning modelling approach (CNN, FCN, DNN, GANs), context (various countries, urban and rural areas), and satellite data (optical, multispectral, radar, all having spatial resolutions of ≤1 m). Notwithstanding that direct comparison with our ResNet models is precluded by this diversity of models and data, as well as the lack of studies specifically for remote semi-forested tropical areas, it is noteworthy that the F1 scores for our ResNet models are greater than or comparable to 11 of the 23 reviewed models for which F1 scores were reported.

The mIoU scores of our three models were comparatively moderate, ranging between 43% for UNet and 58% for ResNet-34 (Figure 9). Our upper mIoU score is equivalent to that of Facebook’s modified D-LinkNet-34 ML road-mapping model when trained on weakly supervised global OSM road data and assessed against the DeepGlobe Challenge reference dataset [50], which spans urban, peri-urban, and rural areas in Indonesia, Thailand, and India. Unsurprisingly, however, our upper mIoU score is less than Facebook’s ultimate ‘finetuned’ model incorporating additional, manually labelled, global OSM road training data, having a mIoU of 64% [36]. Unlike our upper F1 score, our upper mIoU score is not very consistent with those of the ML road-detection models reviewed by Abdallahi et al. [26]. Our upper mIoU score is greater than or comparable to only three [51–53] of the eleven models reviewed for which IoU was reported [32,33,54–57]. In comparison, Facebook’s finetuned road-detection model—a useful referent given its global deployment and public usage [19,58]—would equal or exceed five of these 11 models on the IoU metric.

The discrepancy between our F1 and mIoU scores reflects the fact that the mIoU metric penalizes misclassification relatively severely. In simple terms, when summarized over all images tested for a model, the mIoU presents a measure approaching worst-case scenario model performance, whereas the F1 score presents a measure approaching average performance under general conditions. In this light, it is telling that, of six ML road extraction models applied to the DeepGlobe dataset [57,59–63] as summarized by Das and Chand [63], our REsNet-34 model exceeded all on the F1 score but none on the IoU measure. Likewise, of six other models reviewed by Abdollahi et al. [26] reporting both F1 and IoU scores [32,52,55,56], including two applied to the DeepGlobe dataset, our ResNet-34 model exceeded all but one on the F1 score, and was within 2% of the highest F1 score, while being inferior to all but two on the IoU score. These comparisons are not to suggest model inferiority or superiority per se, but rather highlight the likelihood that our models encountered relatively rare but significant instances of road-detection error. In the specific context of remote semi-forested tropical landscapes, a leading candidate for such error is the failure of models to detect relatively faint, rustic, semi-vegetated roadways, e.g., narrow, irregular dirt tracks traversing dense forest canopy, or faint tracks traversing semi-exposed soil. Another candidate for such error is the occasional misclassification of artificial image edges resultant of image processing, erroneously classified as *Road* by the models (Figure 10a,d,g,i,j). The latter error could be readily avoided by implementing a simple flood-fill algorithm or similar to identify and remove uniform border pixels introduced during image processing (e.g., black borders of input tiles in Figure 10a,d,g,i,j).

The discrepancy between our mIoU and F1 scores is also notable in that it is smaller for the ResNet models than for UNet, proportionally and absolutely (Figure 9). Greater accuracy for the two ResNet models is probably due in part to their greater propensity to capture relatively faint and/or irregular road features. ResNet achieved greater coverage of such road features and thus of roads generally partially by capturing such roads as ‘broken’, ‘spotty’, or thin features in output road maps, compared to the more definite, thicker, but fewer road features output by UNet (Figure 10a,b,e,i,j,l). In other words, greater accuracies of the ResNet models were seemingly achieved partly because of, not in spite of, the relatively disjointed or faint road features in their output classifications.

Remote Sens. 2024, 16, 839
&lt;page_number&gt;12 of 18&lt;/page_number&gt;

<table>
  <thead>
    <tr>
      <th rowspan="2">Image Tile</th>
      <th colspan="1">Reference Data</th>
      <th colspan="2">Model Output</th>
    </tr>
    <tr>
      <th>UNet</th>
      <th>ResNet-34</th>
      <th>ResNet-34 +</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>(a)</td>
      <td>&lt;img&gt;Image tile (a) - Reference Data&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (a) - UNet Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (a) - ResNet-34 Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (a) - ResNet-34 + Output&lt;/img&gt;</td>
    </tr>
    <tr>
      <td>(b)</td>
      <td>&lt;img&gt;Image tile (b) - Reference Data&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (b) - UNet Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (b) - ResNet-34 Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (b) - ResNet-34 + Output&lt;/img&gt;</td>
    </tr>
    <tr>
      <td>(c)</td>
      <td>&lt;img&gt;Image tile (c) - Reference Data&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (c) - UNet Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (c) - ResNet-34 Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (c) - ResNet-34 + Output&lt;/img&gt;</td>
    </tr>
    <tr>
      <td>(d)</td>
      <td>&lt;img&gt;Image tile (d) - Reference Data&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (d) - UNet Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (d) - ResNet-34 Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (d) - ResNet-34 + Output&lt;/img&gt;</td>
    </tr>
    <tr>
      <td>(e)</td>
      <td>&lt;img&gt;Image tile (e) - Reference Data&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (e) - UNet Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (e) - ResNet-34 Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (e) - ResNet-34 + Output&lt;/img&gt;</td>
    </tr>
    <tr>
      <td>(f)</td>
      <td>&lt;img&gt;Image tile (f) - Reference Data&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (f) - UNet Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (f) - ResNet-34 Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (f) - ResNet-34 + Output&lt;/img&gt;</td>
    </tr>
    <tr>
      <td>(g)</td>
      <td>&lt;img&gt;Image tile (g) - Reference Data&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (g) - UNet Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (g) - ResNet-34 Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (g) - ResNet-34 + Output&lt;/img&gt;</td>
    </tr>
    <tr>
      <td>(h)</td>
      <td>&lt;img&gt;Image tile (h) - Reference Data&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (h) - UNet Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (h) - ResNet-34 Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (h) - ResNet-34 + Output&lt;/img&gt;</td>
    </tr>
    <tr>
      <td>(i)</td>
      <td>&lt;img&gt;Image tile (i) - Reference Data&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (i) - UNet Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (i) - ResNet-34 Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (i) - ResNet-34 + Output&lt;/img&gt;</td>
    </tr>
    <tr>
      <td>(j)</td>
      <td>&lt;img&gt;Image tile (j) - Reference Data&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (j) - UNet Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (j) - ResNet-34 Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (j) - ResNet-34 + Output&lt;/img&gt;</td>
    </tr>
    <tr>
      <td>(k)</td>
      <td>&lt;img&gt;Image tile (k) - Reference Data&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (k) - UNet Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (k) - ResNet-34 Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (k) - ResNet-34 + Output&lt;/img&gt;</td>
    </tr>
    <tr>
      <td>(l)</td>
      <td>&lt;img&gt;Image tile (l) - Reference Data&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (l) - UNet Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (l) - ResNet-34 Output&lt;/img&gt;</td>
      <td>&lt;img&gt;Image tile (l) - ResNet-34 + Output&lt;/img&gt;</td>
    </tr>
  </tbody>
</table>

**Figure 10.** A selection of input, reference, and output images showing variation to road-mapping accuracy across models and contexts. Image tiles are the resampled high-resolution satellite images of 256 × 256 pixels. Reference data show the manual delineations of observed road features in each image tile. Model outputs show models’ corresponding predictions of road features.

Remote Sens. 2024, 16, 839
&lt;page_number&gt;13 of 18&lt;/page_number&gt;

## 4. Discussion

This study clarifies the potential of machine learning modelling for automated road mapping across remote semi-forested tropical regions, alongside Botelho et al. [13] for Amazonia. Our minimum F1 score of model accuracy, at 72% for the UNet model based on resampled high-resolution true-color satellite imagery, was only slightly higher than that achieved by Botelho et al. [13] using UNet and medium-resolution multi-spectral Sentinel-2 satellite imagery. On this basis, we postulate that the UNet model may have an upper F1 score of ~70–75% for road mapping in remote, generally forested tropical regions. The greater accuracies of the two ResNet models recommend these models over UNet for in remote tropical regions. Various other modelling approaches would doubtless prove more accurate [36,57], albeit often at the cost of much greater complexity and computational resources [26].

The appreciable accuracies of our models for equatorial Asia-Pacific, as for Botelho et al. [13] in Amazonia and the DeepGlobe Challenge dataset [50] incorporating rural areas [64], support the possibility of a concerted scientific program of autonomous road mapping at very large scales. Ideally, a single program would consistently map all (unmapped) roads pantropically, commencing with relatively environmentally intact areas threatened by road development, to benefit the broader scientific, environmental, and policy communities. In this sense, such a program would resemble numerous earlier applied-science programs that mapped poorly defined environmental dynamics of longstanding concern and whose outputs are now standard reference data (e.g., human footprints and natural areas [65,66], intact vegetation [67,68], deforestation ‘alerts’ [69–72], wildfires [73–76], wilderness areas [77], tree plantations [78–81], and human settlement [82]).

Like these earlier programs, a pantropical road-mapping program would ideally feature coordination between scientific, civil/environmental, and policy interests, and would be based on freely available data and open-source methodologies. Requisite road ‘training’ data, based on visual interpretations of satellite imagery, are already available and possibly sufficient for many major tropical regions, including most of equatorial Asia-Pacific [11,16], the Brazilian Amazon [13,15,21], and tropical Africa [5,22]. In Southeast Asia and Africa, as elsewhere, human-curated road data are available to varying degrees via OSM, with some countries or areas having been extensively mapped in recent years [83,84], although coverage in remote areas is probably relatively sparse [11,16]. It is envisaged that any road-mapping application resultant of such a program could be disseminated as a simple online interface between a given ML model and Google Earth, OSM, or a similar geospatial platform, cf. [85]. We envisage an interface whereby users may visually interpret new roads but also produce updated, ML-generated road maps to monitor any region of interest ongoingly. As demonstrated here, such a program could be based on freely available ‘screenshots’ of high-resolution satellite imagery accessible via Google Earth or similar platforms.

Today, however, an alternative, even contrasting road-mapping program characterized by ‘Big Tech’ and proprietary interests is more likely, if not already realized. Facebook has recently applied proprietary ML algorithms to commercial Maxar high-resolution satellite imagery to map roads globally, reportedly to expand rural internet access and social media activity [18,37]. Other Big Tech interests are following suit for similar commercial reasons [86], e.g., the enhancement of navigational or social apps. Concerns that proprietary data and commercial interests might preclude scientific collaboration and coordination seem at least partially founded. While Facebook has gifted its ML-generated road data to OSM and published tools allowing for users to edit these data [87–89], the underlying ML model remains proprietary, and the underlying commercial satellite imagery is practically unavailable to the scientific community due to its significant cost.

With Facebook Roads data now available globally via the OSM database [37], a collaborative, transparent road-mapping program as envisaged above would potentially be redundant, at least with respect to its outputs. Such a program would however still serve as a rigorous, possibly corrective check of Big Tech mapping, or otherwise fill a ‘niche’ interest

Remote Sens. 2024, 16, 839
&lt;page_number&gt;14 of 18&lt;/page_number&gt;

of the environmental community. The fidelity of Facebook Roads for environmental monitoring of remote tropical areas specifically warrants scrutiny. Facebook’s road-mapping algorithm explicitly excluded road training data for areas with relatively few roads, instead focusing model training on “areas that are more completely mapped” by the OSM database [36]—a practice not uncommon in the literature [33]. Therefore, notwithstanding the massive quantity of the global OSM training data, Facebook Roads may still tend to omit or misclassify the often irregular, partially treed, rustic roads typical of remote tropical areas. A cursory review of Facebook Roads in remote semi-forested regions of Brazil, India, and Panama found various instances of rivers or dry river beds conflated with roads, for example. Pending formal scrutiny of Facebook Roads, this issue of the quality vs. quantity of Facebook Roads will however likely prove of secondary importance to conservation scientists and policymakers who have long awaited any substantive road data in remote areas.

**Supplementary Materials:** The following supporting information can be downloaded at: https://www.mdpi.com/article/10.3390/rs16050839/s1, Figure S1: Training and validation loss for (a) UNet, (b) ResNet-34, and (c) ResNet-34+ models over 30 epochs.

**Author Contributions:** Conceptualization, R.R.T., T.H. and W.F.L.; Methodology, R.R.T. and S.S.; Software, R.R.T. and T.H.; Validation, T.H., R.R.T. and S.S.; Formal Analysis, R.R.T.; Investigation, S.S. and R.R.T.; Resources, S.S. and W.F.L.; Data Curation, R.R.T., S.S. and J.E.; Writing—Original Draft Preparation, S.S. and R.R.T.; Writing—Review and Editing, S.S., T.H., W.F.L. and R.R.T.; Visualization, S.S.; Supervision, T.H.; Project Administration, S.S.; Funding Acquisition, S.S. and W.F.L. All authors have read and agreed to the published version of the manuscript.

**Funding:** James Cook University, a private philanthropic foundation, and a Canada Research Chair from The Canadian Tri-Agency Scientific Funding Body (CRC-2020-305) provided research support.

**Data Availability Statement:** The DOI given by Sloan et al. [45], https://doi.org/10.5061/dryad.bvq83bkg7, provides all input image data for the replication and elaboration of this study, including (1) the 200 input satellite images, and the derived (2) 8904 image tiles, each of which entails a true-color image and a corresponding road-reference image. Image tiles used for model training are separated from those used for validation and testing.

**Acknowledgments:** We thank Yoko Ishida and William Reid for logistical and statistical assistance.

**Conflicts of Interest:** The authors declare no conflicts of interest.

**Notes**
1 The term ‘human-curated road data’ implies data generated by human contributions to the OSM database and excludes road data generated by ML models. The following text notes Facebook Roads road data generated by an ML model and added to OSM.

**References**
1. Dulac, J. *Global Land Transport Infrastructure Requirements: Estimating Road and Railway Infrastructure Capacity and Costs to 2050*; International Energy Agency: Paris, France, 2013.
2. Hettige, H. *When Do Rural Roads Benefit the Poor and How? An In-Depth Analysis*; Asian Development Bank: Manilla, Philippines, 2006.
3. Laurance, W.F.; Goosem, M.; Laurance, S.G.W. Impacts of roads and linear clearings on tropical forests. *Trends Ecol. Evol.* **2009**, *24*, 659–669. [CrossRef]
4. Ascensão, F.; Fahrig, L.; Clevenger, A.P.; Corlett, R.T.; Jaeger, J.A.G.; Laurance, W.F.; Pereira, H.M. Environmental challenges for the Belt and Road Initiative. *Nat. Sustain.* **2018**, *1*, 206–209. [CrossRef]
5. Kleinschroth, F.; Laporte, N.; Laurance, W.F.; Goetz, S.; Ghazoul, J. Road expansion and persistence in forests of the Congo Basin. *Nat. Sustain.* **2019**, *2*, 628–634. [CrossRef]
6. Ibisch, P.L.; Hoffmann, M.T.; Kreft, S.; Pe’er, G.; Kati, V.; Biber-Freudenberger, L.; DellaSala, D.A.; Vale, M.M.; Hobson, P.R.; Selva, N. A global map of roadless areas and their conservation status. *Science* **2016**, *354*, 1423–1427. [CrossRef]
7. Laurance, W.F.; Cochrane, M.A.; Bergen, S.; Fearnside, P.M.; Delamônica, P.; Barber, C.; D’Angelo, S.; Fernandes, T. The Future of the Brazilian Amazon. *Science* **2001**, *291*, 438–439. [CrossRef]
8. Wali, A. The transformation of a frontier: State and regional relationships in Panama, 1972–1990. *Hum. Organ.* **1993**, *52*, 115–129. [CrossRef]

Remote Sens. 2024, 16, 839
&lt;page_number&gt;15 of 18&lt;/page_number&gt;

9. Pfaff, A.; Robalino, J.; Walker, R.; Aldrich, S.; Caldas, M.; Reis, E.; Perz, S.; Bohrer, C.; Arima, E.; Laurance, W.; et al. Road investments, spatial spillovers, and deforestation in the Brazilian Amazon. *J. Reg. Sci.* **2007**, *47*, 109–123. [CrossRef]
10. Barber, C.P.; Cochrane, M.A.; Souza, C.M., Jr.; Laurance, W.F. Roads, deforestation, and the mitigating effect of protected areas in the Amazon. *Biol. Conserv.* **2014**, *177*, 203–209. [CrossRef]
11. Hughes, A.C. Have Indo-Malaysian forests reached the end of the road? *Biol. Conserv.* **2018**, *223*, 129–137. [CrossRef]
12. Souza, C.; Ribeiro, J.G.; Botelho, J.P.J.; Kirchhoff, F.T. Advances on Earth Observation and Artificial Intelligence to Map Unofficial Roads in the Brazilian Amazon Biome. Paper Presented at American Geophysical Union, Fall Meeting, 2020, December. Available online: https://ui.adsabs.harvard.edu/abs/2020AGUFMGC106..09S/abstract (accessed on 5 June 2023).
13. Botelho, J.; Costa, S.C.P.; Ribeiro, J.G.; Souza, C.M. Mapping roads in the Brazilian Amazon with artificial intelligence and Sentinel-2. *Remote Sens.* **2022**, *14*, 3625. [CrossRef]
14. Meijer, J.R.; Huijbregts, M.A.J.; Schotten, K.C.G.J.; Schipper, A.M. Global patterns of current and future road infrastructure. *Environ. Res. Lett.* **2018**, *13*, 064006. [CrossRef]
15. Das Neves, P.B.T.; Blanco, C.J.C.; Montenegro Duarte, A.A.A.; das Neves, F.B.S.; das Neves, I.B.S.; de Paula dos Santos, M.H. Amazon rainforest deforestation influenced by clandestine and regular roadway network. *Land Use Policy* **2021**, *108*, 105510. [CrossRef]
16. Engert, J.; Campbell, M.J.; Cinner, J.; Ishida, Y.; Sloan, S.; Alamgir, M.; Cislowski, J.; Laurance, W.F. 'Ghost roads' and the survival of tropical forests. *Nature* **2024**.
17. Engert, J.E.; Ishida, F.Y.; Laurance, W.F. Rerouting a major Indonesian mining road to spare nature and reduce development costs. *Conserv. Sci. Pract.* **2021**, *3*, e521. [CrossRef]
18. BBC. Facebook Uses AI to Map Thailand's Roads. *BBC News*. 24 July 2019. Available online: https://www.bbc.com/news/technology-49091093 (accessed on 2 January 2023).
19. Cole, L.J. Mapping the World. Pegasus: The Magazine of the University of Central Florida. 2020. Available online: https://www.ucf.edu/pegasus/mapping-the-world/ (accessed on 2 June 2023).
20. Sloan, S.; Campbell, M.J.; Alamgir, M.; Collier-Baker, E.; Nowak, M.; Usher, G.; Laurance, W.F. Infrastructure development and contested forest governance threaten the Leuser Ecosystem, Indonesia. *Land Use Policy* **2018**, *77*, 298–309. [CrossRef]
21. Brandão, A.O.; Souza, C.M. Mapping unofficial roads with Landsat images: A new tool to improve the monitoring of the Brazilian Amazon rainforest. *Int. J. Remote Sens.* **2006**, *27*, 177–189. [CrossRef]
22. Laporte, N.T.; Stabach, J.A.; Grosch, R.; Lin, T.S.; Goetz, S.J. Expansion of industrial logging in Central Africa. *Science* **2007**, *316*, 1451. [CrossRef]
23. Gaveau, D.L.A.; Sloan, S.; Molidena, M.; Yaen, H.; Sheil, D.; Abram, N.K.; Ancrenaz, M.; Nasi, R.; Quinones, M.; Wielaard, N.; et al. Four Decades of Forest Persistence, Clearance and Logging on Borneo. *PLoS ONE* **2014**, *9*, e101654. [CrossRef] [PubMed]
24. Sloan, S.; Alamgir, M.; Campbell, M.J.; Setyawati, T.; Laurance, W.F. Development Corridors and Remnant-Forest Conservation in Sumatra, Indonesia. *Trop. Conserv. Sci.* **2019**, *12*, 194008291988950. [CrossRef]
25. Laurance, W.F.; Achard, F.; Peedell, S.; Schmitt, S. Big data, big opportunities. *Front. Ecol. Environ.* **2016**, *14*, 347. [CrossRef]
26. Abdollahi, A.; Pradhan, B.; Shukla, N.; Chakraborty, S.; Alamri, A. Deep learning approaches applied to remote sensing datasets for road extraction: A state-of-the-art review. *Remote Sens.* **2020**, *12*, 1444. [CrossRef]
27. Gabriele Moser, J.Z. *Mathematical Models for Remote Sensing Image Processing*; Springer: Berlin/Heidelberg, Germany, 2018.
28. Yuan, X.; Shi, J.; Gu, L. A review of deep learning methods for semantic segmentation of remote sensing imagery. *Expert Syst. Appl.* **2021**, *169*, 114417. [CrossRef]
29. Hoeser, T.; Kuenzer, C. Object detection and image segmentation with deep learning on Earth observation data: A review—Part I: Evolution and recent trends. *Remote Sens.* **2020**, *12*, 1667. [CrossRef]
30. Diakogiannis, F.I.; Waldner, F.; Caccetta, P.; Wu, C. ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data. *ISPRS J. Photogramm. Remote Sens.* **2020**, *162*, 94–114. [CrossRef]
31. Alshehhi, R.; Marpu, P.R.; Woon, W.L.; Mura, M.D. Simultaneous extraction of roads and buildings in remote sensing imagery with convolutional neural networks. *ISPRS J. Photogramm. Remote Sens.* **2017**, *130*, 139–149. [CrossRef]
32. Gao, L.; Song, W.; Dai, J.; Chen, Y. Road extraction from high-resolution remote sensing imagery using refined deep residual convolutional neural network. *Remote Sens.* **2019**, *11*, 552. [CrossRef]
33. Buslaev, A.; Seferbekov, S.; Iglovikov, V.; Shvets, A. Fully Convolutional Network for Automatic Road Extraction from Satellite Imagery. In Proceedings of the 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Salt Lake City, UT, USA, 18–22 June 2018; pp. 207–210.
34. Liu, J.; Qin, Q.; Li, J.; Li, Y. Rural road extraction from high-resolution remote sensing images based on geometric feature inference. *ISPRS Int. J. Geo-Inf.* **2017**, *6*, 314. [CrossRef]
35. Dai, J.; Ma, R.; Ai, H. Semi-automatic extraction of rural roads from high-resolution remote sensing images based on a multifeature combination. *IEEE Geosci. Remote Sens. Lett.* **2022**, *19*, 3000605. [CrossRef]
36. Bonafilia, D.; Gill, J.; Basu, S.; Yang, D. Building high resolution maps for humanitarian aid and development with weakly- and semi-supervised learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, Long Beach, CA, USA, 15–20 June 2019; pp. 1–9.

Remote Sens. 2024, 16, 839
&lt;page_number&gt;16 of 18&lt;/page_number&gt;

37. Facebook. Open Mapping at Facebook—A Documentation Repository and Data Host for Facebook’s Mapping-with-AI Project on OpenStreetMap. Available online: https://github.com/facebookmicrosites/Open-Mapping-At-Facebook (accessed on 2 August 2023).
38. Wurm, M.; Stark, T.; Zhu, X.X.; Weigand, M.; Taubenböck, H. Semantic segmentation of slums in satellite images using transfer learning on fully convolutional neural networks. *ISPRS J. Photogramm. Remote Sens.* **2019**, *150*, 59–69. [CrossRef]
39. He, K.; Zhang, X.; Ren, S.; Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 27–30 June 2016; pp. 770–778. Available online: https://ieeexplore.ieee.org/document/7780459 (accessed on 5 August 2023).
40. Alamgir, M.; Sloan, S.; Campbell, M.J.; Laurance, W.F. Regional economic growth initiative challenges sustainable development and forest conservation in Sarawak, Borneo. *PLoS ONE* **2020**, *15*, e0229614. [CrossRef]
41. Alamgir, M.; Sloan, S.; Campbell, M.J.; Engert, J.; Laurance, W.F. Infrastructure expansion projects undermine sustainable development and forest conservation in Papua New Guinea. *PLoS ONE* **2019**, *4*, e0219408. [CrossRef]
42. Alamgir, M.; Campbell, M.J.; Sloan, S.; Suhardiman, A.; Laurance, W.F. High-risk infrastructure projects pose imminent threats to forests in Indonesian Borneo. *Sci. Rep.* **2019**, *9*, 140. [CrossRef]
43. Sloan, S.; Campbell, M.J.; Alamgir, M.; Lechner, A.M.; Engert, J.; Laurance, W.F. Trans-national conservation and infrastructure development in the Heart of Borneo. *PLoS ONE* **2019**, *19*, e0221947. [CrossRef]
44. Sloan, S.; Campbell, M.; Alamgir, M.; Engert, J.; Ishida, F.Y.; Senn, N.; Huther, J.; Laurance, W.F. Hidden challenges for conservation and development along the Papuan economic corridor. *Environ. Sci. Policy* **2019**, *92*, 98–106. [CrossRef]
45. Sloan, S.; Talkhani, R.R.; Huang, T.; Engert, J.; Laurance, W.F. Satellite Images and Road-Reference Images for AI-Based Road Mapping in Equatorial Asia. Datadryad.org. 2023. Available online: https://datadryad.org/stash/dataset/doi:10.5061/dryad.bvq83bkg7 (accessed on 5 August 2023). [CrossRef]
46. Shorten, C.; Khoshgoftaar, T.M. A survey on image data augmentation for deep learning. *J. Big Data* **2019**, *6*, 60. [CrossRef]
47. Ronneberger, O.; Fischer, P.; Brox, T. U-Net: Convolutional networks for biomedical image segmentation. *arXiv* **2015**. [CrossRef]
48. Li, Z.; Liu, F.; Yang, W.; Peng, S.; Zhou, J. A survey of convolutional neural networks: Analysis, applications, and prospects. *IEEE Trans. Neural Netw. Learn. Syst.* **2022**, *33*, 6999–7019. [CrossRef]
49. Lin, Y.; Xu, D.; Wang, N.; Shi, Z.; Chen, Q. Road extraction from very-high-resolution remote sensing images via a nested SE-deeplab model. *Remote Sens.* **2020**, *12*, 2985. [CrossRef]
50. Demir, I.; Koperski, K.; Lindenbaum, D.; Pang, G.; Huang, J.; Basu, S.; Hughes, F.; Tuia, D.; Raskar, R. Deepglobe 2018: A challenge to parse the earth through satellite images. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, Salt Lake City, UT, USA, 18–22 June 2018; pp. 172–181.
51. Henry, C.; Azimi, S.M.; Merkle, N. Road segmentation in SAR satellite images with deep fully convolutional neural networks. *IEEE Geosci. Remote Sens. Lett.* **2018**, *15*, 1867–1871. [CrossRef]
52. He, H.; Yang, D.; Wang, S.; Wang, S.; Liu, X. Road segmentation of cross-modal remote sensing images using deep segmentation network and transfer learning. *Ind. Robot. Int. J. Robot. Res. Appl.* **2019**, *46*, 384–390. [CrossRef]
53. Doshi, J. Residual inception skip network for binary segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, Salt Lake City, UT, USA, 18–22 June 2018; pp. 216–219.
54. Li, Y.; Xu, L.; Rao, J.; Guo, L.; Yan, Z.; Jin, S. A Y-Net deep learning method for road segmentation using high-resolution visible remote sensing images. *Remote Sens.* **2019**, *10*, 381–390. [CrossRef]
55. Xie, Y.; Miao, F.; Zhou, K.; Peng, J. HsgNet: A road extraction network based on global perception of high-order spatial information. *ISPRS Int. J. Geo-Inf.* **2019**, *8*, 571. [CrossRef]
56. Xin, J.; Zhang, X.; Zhang, Z.; Fang, W. Road extraction of high-resolution remote sensing images derived from DenseUNet. *Remote Sens.* **2019**, *11*, 2499. [CrossRef]
57. Zhou, L.; Zhang, C.; Wu, M. D-LinkNet: LinkNet with pretrained encoder and dilated convolution for high resolution satellite imagery road extraction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, Salt Lake City, UT, USA, 18–22 June 2018; pp. 182–186.
58. Draper, C.; Adiatma, D.; Kanyenye, T.J. Reaching Inaccessible Communities Through Road Mapping for Sustainable Development. Available online: https://www.hotosm.org/updates/reaching-inaccessible-communities-through-road-mapping-for-sustainable-development/ (accessed on 3 February 2024).
59. Máttyus, G.; Luo, W.; Urtasun, R. Deeproadmapper: Extracting road topology from aerial images. In Proceedings of the IEEE International Conference on Computer Vision, Venice, Italy, 22–29 October 2017; pp. 3438–3446.
60. Zhang, Z.; Liu, Q.; Wang, Y. Road extraction by deep residual U-Net. *IEEE Geosci. Andremote Sens. Lett.* **2018**, *15*, 749–753. [CrossRef]
61. Xu, Y.; Xie, Z.; Feng, Y.; Chen, Z. Road extraction from high-resolution remote sensing imagery using deep learning. *Remote Sens.* **2018**, *10*, 1461. [CrossRef]
62. Sun, T.; Chen, Z.; Yang, W.; Wang, Y. Stacked U-Nets with multi-output for road extraction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, Salt Lake City, UT, USA, 18–22 June 2018; pp. 202–206.
63. Das, P.; Chand, S. Extracting road maps from high-resolution satellite imagery using refined DSE-LinkNet. *Connect. Sci.* **2021**, *33*, 278–295. [CrossRef]

Remote Sens. 2024, 16, 839
&lt;page_number&gt;17 of 18&lt;/page_number&gt;

64. CVPR. DeepGlobe Road Extraction Challenge > Results. Available online: https://competitions.codalab.org/competitions/18467#results (accessed on 10 August 2023).
65. Venter, O.; Sanderson, E.W.; Magrach, A.; Allan, J.R.; Beher, J.; Jones, K.R.; Possingham, H.P.; Laurance, W.F.; Wood, P.; Fekete, B.M.; et al. Sixteen years of change in the global terrestrial human footprint and implications for biodiversity conservation. *Nat. Commun.* **2016**, *7*, 12558. [CrossRef]
66. Sanderson, E.W.; Jaiteh, M.; Levy, M.A.; Redford, K.H.; Wannebo, A.V.; Woolmer, G. The human footprint and the last of the wild. *BioScience* **2002**, *52*, 891–904. [CrossRef]
67. Potapov, P.; Yaroshenko, A.; Turubanova, S.; Dubinin, M.; Laestadius, L.; Thies, C.; Aksenov, D.; Egorov, A.; Yesipova, Y.; Glushkov, I.; et al. Mapping the world’s intact forest landscapes by remote sensing. *Ecol. Soc.* **2008**, *13*, 51. [CrossRef]
68. Sloan, S.; Jenkins, C.N.; Joppa, L.N.; Gaveau, D.L.A.; Laurance, W.F. Remaining natural vegetation in the global biodiversity hotspots. *Biol. Conserv.* **2014**, *117*, 12–24. [CrossRef]
69. Hansen, M.C.; Krylov, A.; Tyukavina, A.; Potapov, P.V.; Turubanova, S.; Zutta, B.; Ifo, S.; Margono, B.; Stolle, F.; Moore, R. Humid tropical forest disturbance alerts using Landsat data. *Environ. Res. Lett.* **2016**, *11*, 034008. [CrossRef]
70. Shang, R.; Zhu, Z.; Zhang, J.; Qiu, S.; Yang, Z.; Li, T.; Yang, X. Near-real-time monitoring of land disturbance with harmonized Landsats 7–8 and Sentinel-2 data. *Remote Sens. Environ.* **2022**, *278*, 113073. [CrossRef]
71. Reymondin, L.; Jarvis, A.; Perez-Uribe, A.; Touval, J.; Argote, K.; Coca, A.; Rebetez, J.; Guevara, E.; Mulligan, M. *A Methodology for Near Real-Time Monitoring of Habitat Change at Continental Scales Using MODIS-NDVI and TRMM*; Terra-i & the International Centre for Tropical Agriculture (CIAT): Rome, Italy, 2012.
72. Hansen, M.C.; Potapov, P.V.; Moore, R.; Hancher, M.; Turubanova, S.A.; Tyukavina, A.; Thau, D.; Stehman, S.V.; Goetz, S.J.; Loveland, T.R.; et al. High-resolution global maps of 21st-century forest cover change. *Science* **2013**, *342*, 850–853. [CrossRef]
73. Justice, C.O.; Giglio, L.; Korontzi, S.; Owens, J.; Morisette, J.T.; Roy, D.; Descloitres, J.; Alleaume, S.; Petitcolin, F.; Kaufman, Y. The MODIS fire products. *Remote Sens. Environ.* **2002**, *83*, 244–262. [CrossRef]
74. Sloan, S.; Locatelli, B.; Wooster, M.J.; Gaveau, D.L.A. Fire activity in Borneo driven by industrial land conversion and drought during El Niño periods, 1982–2010. *Glob. Environ. Change* **2017**, *47*, 95–109. [CrossRef]
75. Sloan, S.; Tacconi, L.; Cattau, M.E. Fire prevention in managed landscapes: Recent successes and challenges in Indonesia. *Mitig. Adapt. Strateg. Glob. Change* **2021**, *26*, 32. [CrossRef]
76. Sloan, S.; Locatelli, B.; Andela, N.; Cattau, M.; Gaveau, D.L.A.; Tacconi, L. Declining severe fire activity on managed lands in Equatorial Asia. *Commun. Earth Environ.* **2022**, *3*, 207. [CrossRef]
77. Mittermeier, R.A.; Mittermeier, C.G.; Brooks, T.M.; Pilgrim, J.D.; Konstant, W.R.; da Fonseca, G.A.B.; Kormos, C. Wilderness and biodiversity conservation. *Proc. Natl. Acad. Sci. USA* **2003**, *100*, 10309–10313. [CrossRef]
78. Fagan, M.E.; Kim, D.-H.; Settle, W.; Ferry, L.; Drew, J.; Carlson, H.; Slaughter, J.; Schaferbien, J.; Tyukavina, A.; Harris, N.; et al. The expansion of tree plantations across tropical biomes. *Nat. Sustain.* **2022**, *5*, 661–688. [CrossRef]
79. Descals, A.; Wich, S.; Meijaard, E.; Gaveau, D.L.A.; Peedell, S.; Szantoi, Z. High-resolution global map of smallholder and industrial closed-canopy oil palm plantations. *Earth Syst. Sci. Data* **2021**, *13*, 1211–1231. [CrossRef]
80. Harris, N.; Goldman Dow, E.; Gibbes, S. *Spatial Database of Planted Trees (SPT Version 1.0), Technical Note*; World Resources Institute: Washington, DC, USA, 2019; p. 36.
81. Sloan, S.; Meyfroidt, P.; Rudel, T.K.; Bongers, F.; Chazdon Robin, L. The forest transformation: Planted tree cover and regional dynamics of tree gains and losses. *Glob. Environ. Change* **2019**, *59*, 101988. [CrossRef]
82. Pesaresi, M.; Ehrlich, D.; Ferri, S.; Florczyk, A.J.; Freire, S.; Halkia, S.; Julea, A.M.; Kemper, T.; Soille, P.; Syrris, V. *Operating Procedure for the Production of the Global Human Settlement Layer from Landsat Data of the Epochs 1975, 1990, 2000, and 2014*; Publications Office of the European Union: Luxembourg, 2016.
83. Reddiar, I.B.; Osti, M. Quantifying transportation infrastructure pressure on Southeast Asian World Heritage forests. *Biol. Conserv.* **2022**, *270*, 109564. [CrossRef]
84. Neis, P.; Zielstra, D. Recent developments and future trends in volunteered geographic information research: The case of OpenStreetMap. *Future Internet* **2014**, *6*, 76–106. [CrossRef]
85. Bey, A.; Sánchez-Paus Díaz, A.; Maniatis, D.; Marchi, G.; Mollicone, D.; Ricci, S.; Bastin, J.-F.; Moore, R.; Federici, S.; Rezende, M.; et al. Collect Earth: Land use and land cover assessment through augmented visual interpretation. *Remote Sens.* **2016**, *8*, 807. [CrossRef]
86. Krietzberg, I. Big Tech Giants Are Working to Change the Mapping Industry. In *TheStreet*; The Arena Group: New York, NY, USA, 2023.
87. Facebook. Open Mapping at Facebook—Map with AI Self-Service Training Document. Available online: https://github.com/facebookmicrosites/Open-Mapping-At-Facebook/wiki (accessed on 5 August 2023).

Remote Sens. 2024, 16, 839
&lt;page_number&gt;18 of 18&lt;/page_number&gt;

88. KaartGroup. Java OpenStreetMap (JOSM) Map-with-AI PlugIn. Available online: [https://github.com/KaartGroup/JOSM_MapWithAI_plugin](https://github.com/KaartGroup/JOSM_MapWithAI_plugin) (accessed on 5 August 2023).
89. Facebook. RapID (v.2)—An Online Interface for Editing OpenStreetMap, including Facebook Roads Data. Available online: [https://rapideditor.org/](https://rapideditor.org/) (accessed on 6 August 2023).

**Disclaimer/Publisher's Note:** The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.