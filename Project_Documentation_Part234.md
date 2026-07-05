# Project Documentation: EuroSAT Classification Pipeline

*(Note: This document contains Sections 2, 3, and 4 to be appended to your existing Project Planning & Management section, as per the documentation guidelines.)*

---

## 2. Literature Review

### Feedback & Evaluation
- **Lecturer’s Assessment (Placeholder/Example):** The project demonstrates a highly systematic approach to deep learning. The exhaustive Exploratory Data Analysis (EDA) on both standard RGB and 13-band multispectral data provides excellent context. The comparative modeling strategy—utilizing transfer learning for RGB and a custom-built ResNet-50 architecture tailored for 13-band TIF data—highlights strong architectural understanding.
- **Suggested Improvements:** 
  - *Cross-Validation:* Implement k-fold cross-validation instead of a single static train/val/test split to yield more robust performance metrics.
  - *Advanced Architectures:* Explore Vision Transformers (ViTs) alongside ResNet to compare CNN performance against self-attention mechanisms.
  - *Interactive Demonstration:* Integrate a lightweight web framework (e.g., Streamlit or Gradio) to allow users to upload custom satellite images for real-time model inference.
- **Final Grading Criteria (Based on Syllabus):** 
  - Documentation & Requirements Analysis (20%)
  - Implementation, Code Quality & Best Practices (40%)
  - System Testing, Evaluation & Accuracy Metrics (20%)
  - Final Presentation & Demo (20%)

---

## 3. Requirements Gathering

### Stakeholder Analysis
- **Data Scientists / Machine Learning Engineers:** Require well-structured, clean, and validated datasets. They need reproducible code pipelines that handle large volumes of data efficiently (via batching) without memory overflow.
- **Environmental Researchers / Urban Planners:** Benefit from the final outputs of the system. They require high-accuracy classification to monitor deforestation, track urbanization, and evaluate crop health.
- **Lecturers / Assessors:** Require transparent, well-commented code, logical system design, and comprehensive reports (e.g., confusion matrices, F1-scores) to fairly evaluate the project's success.

### User Stories & Use Cases
- **User Story 1:** As a researcher, I want to visualize the pixel intensity distributions of both RGB channels and Multispectral bands so that I can understand the underlying properties of different land-use categories.
- **User Story 2:** As a data scientist, I want to train a custom ResNet-50 model specifically on 13-band TIF data to determine if multispectral data yields better classification accuracy than standard RGB imagery.
- **User Story 3:** As an assessor, I want the system to output confusion matrices and classification reports so that I can identify precisely which land-cover classes the model struggles to differentiate.

### Functional Requirements
1. **Data Ingestion & Validation:** The system must automatically detect the environment (Kaggle vs. Local), load RGB (.jpg) and Multispectral (.tif) images, and check for corrupted files, missing labels, and duplicate hashes.
2. **EDA Generation:** The system must generate visual analytics, including class distributions, channel histograms, HSV spatial analysis, and spectral signatures for 13 bands.
3. **Data Pipelines:** The system must utilize `tf.data` generators to load images in batches (e.g., 32 images per batch), applying dynamic data augmentation to the training sets to prevent overfitting.
4. **Model Architecture:** The system must support transfer learning via a pretrained ResNet-50 model (for RGB) and compile a custom 13-channel ResNet-50 from scratch (for TIF).
5. **Evaluation:** The system must evaluate the models on an unseen test set and output accuracy, precision, recall, F1-scores, and plotted learning curves.

### Non-functional Requirements
- **Performance:** Training algorithms must be optimized for GPU acceleration. Data loading should utilize background prefetching to ensure the GPU is never idle waiting for data.
- **Reliability:** The data pipeline must handle exceptions gracefully, skipping unreadable or corrupted files rather than crashing the execution.
- **Reproducibility:** Global random seeds (e.g., Seed: 42) must be strictly enforced across NumPy, Python, and TensorFlow to guarantee identical results across runs.
- **Maintainability & Usability:** The codebase must be highly modular ("Chunks") and heavily commented, adhering to industry-standard naming conventions.

---

## 4. System Analysis & Design

### 1. Problem Statement & Objectives
- **Problem Statement:** Classifying land cover from satellite imagery is a critical task for global environmental monitoring. However, traditional manual analysis is slow, and standard machine learning methods struggle with the sheer scale of the data and the high dimensionality of multispectral imagery.
- **Objectives:** 
  1. Build an automated deep learning pipeline capable of classifying 10 distinct land-use categories using the EuroSAT dataset.
  2. Perform rigorous EDA to understand the statistical distribution of the dataset.
  3. Compare the classification efficacy of standard RGB imagery against 13-band multispectral data.
- **Use Case Descriptions (Logical Actor Flow):**
  - *Actor:* System Operator. 
  - *Action:* Executes the pipeline scripts. 
  - *System Response:* The system automatically locates data paths, validates splits, trains the required neural networks, monitors validation loss to trigger early stopping, and dumps evaluation artifacts to the working directory.
- **Software Architecture:** The system adopts a **Sequential Data Processing Pipeline** architecture. It transitions from Data Validation -> Exploratory Data Analysis -> `tf.data` Pipeline Construction -> Model Compilation -> Training Loop -> Evaluation & Reporting.

### 2. Database Design & Data Modeling (Logical Approach)
*Note: Because this is a Machine Learning pipeline, it relies on an organized flat-file data lake rather than a traditional relational database (RDBMS).*
- **Logical Schema:**
  - **Dataset Lake:** Organized hierarchically by classes. `EuroSAT/` (RGB) and `EuroSATallBands/` (TIF), each containing 10 subdirectories (e.g., `Forest`, `Highway`).
  - **Metadata Tables:** `train.csv`, `validation.csv`, and `test.csv` acting as relational tables linking the `Filename` (Primary Key) to the `ClassName` and integer `Label`.
  - **Artifact Storage:** Output weights are stored logically as `.keras` checkpoints. Metrics are maintained in-memory as history dictionaries and saved logically as visualization plots.

### 3. Data Flow & System Behavior (Logical Perspectives)
- **Data Flow Diagram (DFD):** 
  - Raw images and CSV tables flow into the `Data Validation Module`.
  - Validated file paths flow into the `Data Generators`, where they are transformed into mathematical Tensors (normalized arrays).
  - Tensors flow through the deep neural network (ResNet) to output classification probabilities. 
  - Ground truth labels and probabilities flow into the `Evaluation Module` to compute the final loss and metrics.
- **Sequence Diagram:** 
  - Operator -> Pipeline Script -> Environment Auto-detector -> Data Loader -> Model Builder -> GPU Training Engine -> Evaluator -> Operator.
- **Activity Diagram:** 
  - *Start* -> Check Environment -> Validate Data Integrity -> Generate EDA -> Construct `tf.data.Dataset` -> Initialize Weights -> Train Epochs -> (Does Validation Loss Improve? If No, trigger Early Stopping) -> Evaluate on Test Set -> Save Results -> *End*.
- **Class Diagram (Conceptual):** 
  - Although written procedurally, the logical objects include: `DatasetManager` (attributes: paths, splits; methods: load_csv, hash_check), `EDAVisualizer` (methods: plot_histograms, plot_spectral_signatures), `ModelArchitect` (methods: build_rgb_resnet, build_tif_resnet), and `Evaluator` (methods: plot_learning_curves, generate_classification_report).

### 4. UI/UX Design & Prototyping
- **UI/UX Guidelines (Console & Notebook UI):** 
  - As a data pipeline, the primary UI is the standard output (terminal/notebook output) and generated figures.
  - **Visual Prototyping:** Matplotlib and Seaborn are used with the `darkgrid` style. Custom color mapping is used so that the 10 distinct classes maintain the same assigned color across all bar charts, pie charts, and scatter plots.
  - **Feedback UI:** Print statements are structured with clear headers, horizontal dividers (`====`), and emojis (✅, ⚠️, 📊, ⏱️) to allow the user to easily scan the logs, understand the pipeline's progress, and immediately notice warnings (e.g., missing data or GPU absence).

### 5. System Deployment & Integration
- **Technology Stack:**
  - *Language:* Python 3
  - *Deep Learning Core:* TensorFlow & Keras
  - *Data Manipulation:* Pandas, NumPy
  - *Geospatial Data Processing:* Rasterio (for 13-band TIF ingestion)
  - *Visualization:* Matplotlib, Seaborn
- **Deployment Diagram (Logical):** 
  - The software is intended to be deployed on a cloud notebook environment (e.g., Kaggle or Google Colab).
  - *Hardware Node:* Uses a CPU for data augmentation and a GPU (e.g., NVIDIA T4 or P100) for matrix multiplication and backpropagation.
  - *Storage Node:* The dataset is mounted as a read-only input drive, while the model checkpoints and generated plots are directed to an ephemeral read-write working directory.
- **Component Diagram:** 
  - The pipeline comprises four main components: 
    1) The **Ingestion Engine** (handles OS paths, CSV parsing, Rasterio TIF reads).
    2) The **Transformation Engine** (handles min-max normalization, ImageNet standardization, augmentation).
    3) The **Machine Learning Core** (contains the ResNet-50 computational graphs).
    4) The **Reporting Engine** (SciKit-Learn metrics calculator and Plotly/Matplotlib image generator).
