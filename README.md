<h2>TensorFlow-FlexUNet-Image-Segmentation-COVID-19-Infection (2025/08/03)</h2>

This is the first experiment of Image Segmentation for COVID-19 Infection Multiclass,
 based on our 
TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) 
and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1LVQ6rnNO9Wpenqjh9_R7k3b6w_sZqyxC/view?usp=sharing">
Augmented-COVID-19-ImageMask-Dataset.zip</a>.
which was derived by us from 
<a href="https://medicalsegmentation.com/covid19/">
<b>
COVID-19 <br> COVID-19 CT segmentation dataset
</b>
</a>
<br>
<br>
<b>Acutual Image Segmentation for 512x512 COVID-19 Infection images</b><br>

As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks, but lack precision in some areas, 
<br>
The green represents a right lung, the blue a left lung, and the red an infection region
respectively.<br>

<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/images/20.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/masks/20.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test_output/20.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/images/barrdistorted_1002_0.3_0.3_76.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/masks/barrdistorted_1002_0.3_0.3_76.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test_output/barrdistorted_1002_0.3_0.3_76.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/images/barrdistorted_1001_0.3_0.3_30.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/masks/barrdistorted_1001_0.3_0.3_30.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test_output/barrdistorted_1001_0.3_0.3_30.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here has been taken from the web-site:<br>

<a href="https://medicalsegmentation.com/covid19/">
<b>
COVID-19 <br> COVID-19 CT segmentation dataset
</b>
</a>
<br>
<br>
This is a dataset of 100 axial CT images from >40 patients with COVID-19 that were converted from openly accessible JPG images found HERE. The conversion process is described in detail in the following blogpost: Covid-19 radiology — data collection and preparation for Artificial Intelligence
<br>
In short, the images were segmented by a radiologist using 3 labels: ground-glass (mask value =1), consolidation (=2) and pleural effusion (=3). We then trained a 2d multilabel U-Net model, which you can find and apply in MedSeg. NEW (from 2nd April 2020): Try the Beta fully automated report HERE (DICOM only).
<br>
Download data:<br>
Training images as .nii.gz (151.8 Mb) – 100 slices<br>
Training masks as .nii.gz (1.4 Mb) – 100 masks<br>
CSV file connecting slice no. with SIRM case no. (0.001 Mb)<br>
Test images as .nii.gz (14.2 Mb) – 10 slices. Kaggle competition<br>

<br>
Comments<br>
Ground-glass opacities have been shown to precede consolidations. Some reports have shown that 
early and more pronounced findings on CT correlate negatively with prognosis. 
Maybe volumetric measurements and ratio of ground-glass/consolidation can further 
enhance the prognosis estimation for COVID-19 patients. However, as radiologists, we feel it is our obligation to mention that CT should generally NOT be used for broad screening for COVID-19 as a substitute for RT-PCR.
<br>
You may use the segmentations freely, but we would very much appreciate an acknowledgment/link.

<br>
<br>


<h3>
<a id="2">
2 COVID-19 ImageMask Dataset
</a>
</h3>
 If you would like to train this COVID-19 Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1LVQ6rnNO9Wpenqjh9_R7k3b6w_sZqyxC/view?usp=sharing">
Augmented-COVID-19-ImageMask-Dataset.zip</a>.
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─COVID-19
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>COVID-19 Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/COVID-19/COVID-19_Statistics.png" width="512" height="auto"><br>
<br>
<!--
On the derivation of the dataset, please refer to the following Python scripts:<br>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
<br>
-->
As shown above, the number of images of train and valid datasets is not so large to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/COVID-19/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/COVID-19/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained COVID-19 TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/COVID-19/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/COVID-19 and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 4

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for COVID-19 1+3 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
; 1+3 classes
; categories  = ["right lung", "left lung", "infection"]
; RGB colors         right_lung:green,left_lung:blue,  infection:red     
rgb_map = {(0,0,0):0,(0,255,0):1,     (0,0, 255):2,    (255, 0, 0):3, }


</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInfereuncer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/COVID-19/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 16,17,18)</b><br>
<img src="./projects/TensorFlowFlexUNet/COVID-19/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 34,35,36)</b><br>
<img src="./projects/TensorFlowFlexUNet/COVID-19/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was terminated at epoch 36.<br><br>
<img src="./projects/TensorFlowFlexUNet/COVID-19/asset/train_console_output_at_epoch36.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/COVID-19/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/COVID-19/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/COVID-19/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/COVID-19/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/COVID-19</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for COVID-19.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/COVID-19/asset/evaluate_console_output_at_epoch36.png" width="920" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/COVID-19/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this COVID-19/test was low and dice_coef_multiclass 
high as shown below.
<br>
<pre>
categorical_crossentropy,0.0895
dice_coef_multiclass,0.9529
</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/COVID-19</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for COVID-19.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/COVID-19/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/COVID-19/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/COVID-19/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels</b><br>
The green represents a consep, the yellow a crag, the cyan a dpath, the red a glas, and the blue a pannuke respectively.<br><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/images/20.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/masks/20.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test_output/20.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/images/barrdistorted_1001_0.3_0.3_77.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/masks/barrdistorted_1001_0.3_0.3_77.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test_output/barrdistorted_1001_0.3_0.3_77.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/images/barrdistorted_1001_0.3_0.3_30.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/masks/barrdistorted_1001_0.3_0.3_30.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test_output/barrdistorted_1001_0.3_0.3_30.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/images/barrdistorted_1005_0.3_0.3_52.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/masks/barrdistorted_1005_0.3_0.3_52.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test_output/barrdistorted_1005_0.3_0.3_52.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/images/barrdistorted_1004_0.3_0.3_20.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/masks/barrdistorted_1004_0.3_0.3_20.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test_output/barrdistorted_1004_0.3_0.3_20.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/images/barrdistorted_1004_0.3_0.3_27.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test/masks/barrdistorted_1004_0.3_0.3_27.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/COVID-19/mini_test_output/barrdistorted_1004_0.3_0.3_27.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>

<b>1. COVID-19 infection segmentation using hybrid deep learning and image processing techniques</b>
Samar Antar, Hussein Karam Hussein Abd El-Sattar, Mohammad H. Abdel-Rahman & Fayed F. M. Ghaleb  <br>

<a href="https://www.nature.com/articles/s41598-023-49337-1">
https://www.nature.com/articles/s41598-023-49337-1
</a>

<br>
<br>
<b>2. COVID-19 Infection Segmentation and Severity Assessment Using a Self-Supervised Learning Approach</b>
<br>
 Yao Song,Jun Liu,Xinghua Liu  and Jinshan Tang<br>
<a href="https://www.mdpi.com/2075-4418/12/8/1805">
https://www.mdpi.com/2075-4418/12/8/1805
</a>
<br>
<br>

<b>3. COVID-19 lung infection segmentation with a novel two-stage cross-domain transfer learning framework</b>
<br>
Jiannan Liu, Bo Dong, Shuai Wang, Hui Cui, Deng-Ping Fan, Jiquan Ma, Geng Chen <br>

<a href="https://www.sciencedirect.com/science/article/pii/S1361841521002504">
https://www.sciencedirect.com/science/article/pii/S1361841521002504
</a>
<br>
<br>

<b>4. Inf-Net: Automatic COVID-19 Lung Infection Segmentation From CT Images</b><br>
Deng-Ping Fan, Tao Zhou, Ge-Peng Ji, Yi Zhou, Geng Chen, Huazhu Fu, Jianbing Shen, Ling Shao<br>
<a href="https://ieeexplore.ieee.org/document/9098956">
https://ieeexplore.ieee.org/document/9098956
</a>

