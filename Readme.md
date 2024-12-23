# Parkinson's Disease Detection using OpenCV

*Team members: Priyanka, Ui-Joon, Seung-Hyuk, Su-Bin*

#### This project implements a system to detect Parkinson's Disease (PD) based on spiral or wave drawings provided by individuals. The system uses machine learning algorithms to classify images into healthy or parkinson categories. The detection process involves real-time video input, preprocessing, feature extraction, and classification. OpenCV is used extensively for image processing, and a Random Forest classifier is employed for robust and interpretable predictions.


## Features ##

* **Real-Time Detection**: Capture and classify input from a live video feed.

* **Paper Detection**: Automatically detects and extracts the region of interest (spiral or wave drawing) from the input.

* **Feature Extraction**: Utilizes Histogram of Oriented Gradients (HOG) and Local Binary Patterns (LBP) for extracting structural and texture features.

* **Preprocessing Techniques**:
    * ***Gamma Correction*** for brightness adjustment.
    * ***Gaussian Blur*** for noise reduction.
    * ***CLAHE*** (Contrast Limited Adaptive Histogram Equalization) for enhanced contrast.

* **Data Augmentation**: Rotates images to artificially expand the dataset and improve generalization.

* **Machine Learning**: Random Forest classifier with hyperparameter tuning for optimal performance.

* **Reference Comparison**: Matches features between input and predefined healthy/unhealthy references for visual evaluation.

* **Cross-Validation**: Validates the model’s performance using stratified k-fold cross-validation.

## Requirements ##

* Python (Version 3.10>)

* OpenCV
```cmd
pip install opencv-python
pip install opencv-contrib-python
```

* Scikit-learn

```cmd
pip install scikit-learn
```

* Scikit-image

```cmd
pip install scikit-image
```

* NumPy

```cmd 
pip install numpy
```

## Dataset Structure

                                             [Drawings]
                                                 |
                              +---------------------+---------------------+
                         [Spriral]                                     [Wave]
                              |                                           |
                   +----------+----------+                     +----------+----------+
             [training]             [testing]             [training]             [testing]
                 |                      |                      |                     |           
          +------+------+       +------+------+         +------+------+       +------+------+
          |             |       |             |         |             |       |             |
      [healthy]  [parkinson] [healthy] [parkinson]  [healthy]  [parkinson]  [healthy]  [parkinson]
                        



## Model Structure
                 [Load & Preprocess Data]
                           |
                 +---------+---------+
        [Extract Features from Original]   [Extract Features from Augmented]
                 |                           |
                 +--------+------------------+
                          |
                    [Combine Data]
                          |
                    [Shuffle Data]
                          |
                    [Encode Labels]
                          |
           +-------------+-------------+
     [Define & Train Random Forest Model]
                          |
                  [Evaluate Accuracy]
                          |
                   [Final Model]


## Results

* Training Accuracy: ~100%

* Testing Accuracy: ~81.67%

## Future Improvements

* Incorporate advanced models like XGBoost or LightGBM for better performance.

* Use larger datasets to improve generalization and reduce overfitting.

* Integrate a GUI for better user interaction.



#### Acknowledgements: Special thanks to the open-source community for libraries like OpenCV, Scikit-learn, and Scikit-image