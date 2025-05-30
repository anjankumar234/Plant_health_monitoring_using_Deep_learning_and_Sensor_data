# Plant Health Monitoring using CNN Features + Sensor Data with Random Forest

This project presents a hybrid plant health monitoring pipeline that combines deep learning-based image feature extraction with sensor data analysis to classify plant health status as Healthy or Unhealthy. The final classifier is a Random Forest model optimized through RandomizedSearchCV.

## Dataset
### 1. Image Data
Located at: Dataset_for_training/train/

Format: Organized in subfolders Healthy/ and Unhealthy/

Naming Convention: {Plant_ID}_{week_no}_{image_id}.jpg

### 2. Sensor Data
CSV: sensor_aggregated_labeled.csv

Columns: Plant_ID, week no, temperature, humidity, light, soil moisture, etc., and label

## Methodology
### 1. Image Feature Extraction
We extract deep features from plant images using the following pretrained CNN models:

EfficientNetB0

ResNet50

MobileNetV2

Each image is preprocessed according to the corresponding model's requirements, and features are extracted from the global average pooling layer.

### 2. Feature Fusion
The outputs of the three CNN models are concatenated.

These are fused with corresponding sensor readings (based on Plant_ID and week no).

### 3. Random Forest Classifier
The concatenated features are used to train a Random Forest.

Hyperparameter tuning is performed using RandomizedSearchCV.

### 4. Evaluation
Evaluated using Accuracy, Precision, Recall, F1-score.

Also provides week-wise breakdown and visual plots.

## Output Plots
weekwise_accuracy_plot.png: Bar chart showing accuracy across weeks.

true_weekly_health_distribution.png: Ground truth distribution of health status per week.

## Sample Prediction
Use the following function to predict plant health on new image + sensor data:


      predict_sample(image_path, plant_id, week_no, sensor_df, best_rf_model)
      Example:
      python
      Copy
      Edit
      sample_image_path = "test/Healthy/C2_10_1.jpg"
      predict_sample(sample_image_path, "C2_10", 1, sensor_df, best_rf_model)
## Model Export
The trained Random Forest model is saved as:

bash
Copy
Edit
best_rf_model.pkl
Use joblib.load("best_rf_model.pkl") to reload the model.

## Dependencies

       pip install numpy pandas scikit-learn tensorflow matplotlib tqdm joblib
       
Also ensure access to:

       EfficientNetB0, ResNet50, MobileNetV2 from Keras applications

Google Drive (if loading from Colab)

## Folder Structure

          ├── Dataset_for_training/
          │   ├── train/
          │   │   ├── Healthy/
          │   │   └── Unhealthy/
          │   └── test/
          ├── sensor_aggregated_labeled.csv
          ├── best_rf_model.pkl
          ├── weekwise_accuracy_plot.png
          ├── true_weekly_health_distribution.png
          ├── main.py  # (Your full training and evaluation code)
          └── README.md
## Results Summary
Overall Accuracy: 98.9%

Best Week Performance: Week Y with accuracy Z.ZZ

Class-wise Performance: Refer to classification report in output


## TODO
 Integrate LSTM for temporal analysis

 Add real-time prediction demo


## License
This project is open-sourced under the MIT License.
