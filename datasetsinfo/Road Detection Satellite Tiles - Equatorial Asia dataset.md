# Satellite images and road-reference data for AI-based road mapping in Equatorial Asia

[https://doi.org/10.5061/dryad.bvq83bkg7](https://doi.org/10.5061/dryad.bvq83bkg7)

**1. INTRODUCTION**
For the purposes of training AI-based models to identify (map) road features in rural/remote tropical regions on the basis of true-colour satellite imagery, and subsequently testing the accuracy of these AI-derived road maps, we produced a dataset of 8904 satellite image ‘tiles’ and their corresponding known road features across Equatorial Asia (Indonesia, Malaysia, Papua New Guinea).
 
**2. FURTHER INFORMATION**
The following is a summary of our data.  Fuller details on these data and their underlying methodology are given in the corresponding article, cited below:
 
Sloan, S., Talkhani, R.R., Huang, T., Engert, J., Laurance, W.F. (2023) Mapping remote roads using artificial intelligence and satellite imagery. *Remote Sensing*. 16(5): 839. [https://doi.org/10.3390/rs16050839](https://doi.org/10.3390/rs16050839)
 
Correspondence regarding these data can be directed to:
 
Sean Sloan
Department of Geography, Vancouver Island University, Nanaimo, B.C, Canada
[sean.sloan@viu.ca](mailto:sean.sloan@viu.ca);  
 
Tao (Kevin) Huang
College of Science and Engineering, James Cook University, Cairns, Queensland 4878, Australia
[tao.huang1@jcu.edu.au](mailto:tao.huang1@jcu.edu.au)
 
**3. INPUT 200 SATELLITE IMAGES**
The main dataset shared here was derived form a set of 200 input satellite images, also provided here.  These 200 images are effectively ‘screenshots’ (i.e., reduced-resolution copies) of high-resolution true-colour satellite imagery (~0.5-1m pixel resolution) observed using the Elvis Elevation and Depth spatial data portal ([https://elevation.fsdf.org.au/](https://elevation.fsdf.org.au/), which here is functionally equivalent to the more familiar Google Earth.  Each of these original images were initially acquired at a resolution of 1920x886 pixels.  Actual image resolution was coarser than the native high-resolution imagery.  Visual inspection of these 200 images suggests a pixel resolution of ~5 meters, given the number of pixels required to span features of familiar scale, such as roads and roofs, as well as the ready discrimination of specific land uses, vegetation types, etc.  These 200 images generally spanned either forest-agricultural mosaics or intact forest landscapes with limited human intervention.   Sloan et al. (2023) presents a map indicating the various areas of Equatorial Asia from which these images were sourced.   
 
**3.1 IMAGE NAMING CONVENTION**
A common naming convention applies to the file names of these 200 satellite images:
 
**XX##.png**

where,
 
·        **XX** – denotes the geographical region / major island of Equatorial Asia of the image, as follows: ‘bo’ (Borneo), ‘su’ (Sumatra), ‘sl’ (Sulawesi), ‘pn’ (Papua New Guinea), ‘jv’ (java), ‘ng’ (New Guinea [i.e., Papua and West Papua provinces of Indonesia])
·        **##** – denotes the i<sup>th</sup> image for a given geographical region / major island amongst the original 200 images, e.g., bo1, bo2, bo3…
 
**4. INTERPRETING ROAD FEATURES IN THE IMAGES**
For each of the 200 input satellite images, its roads were visually interpreted and manually digitized to create a reference image dataset by which to train, validate, and test AI road-mapping models, as detailed in Sloan et al. (2023).  The reference dataset of road features was digitized using the ‘pen tool’ in Adobe Photoshop.  The pen’s ‘width’ was held constant over varying scales of observation (i.e., image ‘zoom’) during digitization.  Consequently, at relatively small scales at least, digitized road features likely incorporate vegetation immediately bordering roads.  The resultant binary (*Road* / *Not Road*) reference images were saved as PNG images with the same image dimensions as the original 200 images.  
 
**5. IMAGE TILES AND REFERENCE DATA FOR MODEL DEVELOPMENT**
The 200 satellite images and the corresponding 200 road-reference images were both subdivided (aka ‘sliced’) into thousands of smaller image ‘tiles’ of 256x256 pixels each.  Subsequent to image subdivision, subdivided images were also rotated by 90, 180, or 270 degrees to create additional, complementary image tiles for model development.  In total, 8904 image tiles resulted from image subdivision and rotation.  These 8904 image tiles are main data of interest disseminated here.  Each image tile entails the true-colour satellite image (256x256 pixels) and a corresponding binary road reference image (*Road* / *Not Road*). 
 
Of these 8904 image tiles, Sloan et al. (2023) randomly selected 80% for model training (during which an AI model ‘learns’ to recognize road features in the input imagery), 10% for model validation (during which model parameters are iteratively refined), and 10% for final model testing (during which the final accuracy of the output road map is assessed).  Here we present these data in two folders accordingly:

*   ‘Training’ – contains 7124 image tiles used for model training in Sloan et al. (2023), i.e., 80% of the original pool of 8904 image tiles.
*   ‘Testing’– contains 1780 image tiles used for model validation and model testing in Sloan et al. (2023), i.e., 20% of the original pool of 8904 image tiles, being the combined set of image tiles for model validation and testing in Sloan et al. (2023).

**5.1 IMAGE TILE NAMING CONVENTION**
A common naming convention applies to the 8904 image tiles’ directories and file names, in both the ‘training’ and ‘testing’ folders:
 
**XX##_A_B_C_D*rotDDD***

where,
 
·        **XX** – denotes the geographical region / major island of Equatorial Asia of the original input 1920x886 pixel image, as follows: ‘bo’ (Borneo), ‘su’ (Sumatra), ‘sl’ (Sulawesi), ‘pn’ (Papua New Guinea), ‘jv’ (java), ‘ng’ (New Guinea [i.e., Papua and West Papua provinces of Indonesia])
·        **##** – denotes the i<sup>th</sup> image for a given geographical region / major island amongst the original 200 images, e.g., bo1, bo2, bo3…
·        **A, B, C and D** – can all be ignored. These values, which are one of 0, 256, 512, 768, 1024, 1280, 1536, and 1792, are effectively ‘pixel coordinates’ in the corresponding original 1920x886-pixel input image.  They were recorded within the names of image tiles’ sub-directories and file names merely to ensure that names/directory were uniquely named)
·        ***rot*** – implies an image rotation. Not all image tiles are rotated, so ‘*rot*’ will appear only occasionally.
·        ***DDD*** – denotes the degree of image-tile rotation, e.g., 90, 180, 270. Not all image tiles are rotated, so ‘*DDD*’ will appear only occasionally.
 
Note that the designator ‘XX##’ is directly equivalent to the filenames of the corresponding 1920x886-pixel input satellite images, detailed above.  Therefore, each image tiles can be ‘matched’ with its parent full-scale satellite image.  For example, in the ‘training’ folder, the subdirectory ‘Bo12_0_0_256_256’ indicates that its image tile therein (also named ‘Bo12_0_0_256_256’) would have been sourced from the full-scale image ‘Bo12.png’.
