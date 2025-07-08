# 🚀 Bank Term Deposit Prediction

Ever wondered how banks pinpoint potential customers for their term deposits? This project dives deep into the world of predictive analytics to answer just that! We're building a powerful **Neural Network** model to forecast whether a client will subscribe to a term deposit, leveraging a real-world banking dataset. Our goal? To empower banks with smarter, more efficient marketing campaigns and optimized resource allocation.

## 🎯 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Deep Dive](#-dataset-deep-dive)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technical Deep Dive](#-technical-deep-dive)
  - [Data Processing](#data-processing)
  - [Feature Engineering](#feature-engineering)
  - [Feature Selection](#feature-selection)
  - [Model Development](#model-development)
- [Results and Analysis](#-results-and-analysis)
- [Rationale for Model Selection](#-rationale-for-model-selection)
- [Conclusion & Future Work](#-conclusion--future-work)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Project Overview

In the competitive banking landscape, targeted marketing is key. This project harnesses historical campaign data to forge a robust predictive model. By meticulously analyzing customer demographics, past interactions, and a wealth of other attributes, our **Neural Network** model aims to predict the likelihood of a customer saying "yes!" to a term deposit. Imagine the impact: significantly improved efficiency and effectiveness in marketing efforts, leading to higher conversion rates and happier customers.

## 📊 Dataset Deep Dive

Our predictive journey begins with the renowned "Bank Marketing" dataset, a treasure trove of information publicly available from the UCI Machine Learning Repository. This dataset chronicles direct marketing campaigns (primarily phone calls) conducted by a Portuguese banking institution. The ultimate quest? To predict if a client will subscribe to a term deposit (our target variable: `y`).

Key features within the dataset include:
- **Client Demographics:** Age, job, marital status, education, default status, housing, and loan details.
- **Last Contact Information:** Type of communication, month, day of week, and call duration.
- **Campaign Specifics:** Number of contacts during this campaign, days since last contact, number of contacts performed before this campaign, and outcome of the previous campaign.
- **Socio-Economic Context:** Euribor 3 month rate, employment variation rate, consumer price index, consumer confidence index, and number of employees.
- **The Golden Target:** `y` - did the client subscribe to a term deposit? (binary: 'yes' or 'no').

Find the raw data files nestled in the `data/` directory:
- `bank-additional-full.csv`: The comprehensive dataset.
- `bank-additional.csv`: A smaller, more manageable version.
- `bank-additional-names.txt`: Your guide to understanding each attribute.

## 📁 Project Structure

Our project is neatly organized for clarity and ease of navigation:

```
.
├── data/                 # 📊 Raw and processed datasets
│   ├── bank-additional-full.csv
│   ├── bank-additional-names.txt
│   └── bank-additional.csv
├── notebooks/            # 🧠 Jupyter notebooks for exploration, modeling, and evaluation
│   └── ML-Final.ipynb
├── docs/                 # 📄 Project documentation, reports, and presentations
│   ├── ML-Final.pdf
│   └── Rationale for Model Selection.docx
├── .gitignore            # 🚫 Files and directories to be ignored by Git
├── LICENSE               # ⚖️ Project licensing information
├── README.md             # 📖 You are here! Project overview and guide
└── requirements.txt      # 📦 Python dependencies for the project
```

## ⚡ Quick Start

Ready to dive in? Follow these steps to get the project up and running on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/Bank-Term-Deposit-Prediction.git
    cd Bank-Term-Deposit-Prediction
    ```
    *(Remember to replace `YOUR_USERNAME` with your GitHub username after creating your repository on GitHub!)*

2.  **Set up your environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Launch the Jupyter Notebook:**
    ```bash
    jupyter notebook notebooks/ML-Final.ipynb
    ```
    Explore the `ML-Final.ipynb` notebook to see the magic unfold!

## 🛠️ Installation

For a more detailed setup, refer to the Quick Start section above. Essentially, you'll need Python 3.x, `pip`, and then install the dependencies listed in `requirements.txt`.

## 💻 Usage

Once your environment is set up, the `notebooks/ML-Final.ipynb` is your primary playground. It walks you through the entire machine learning pipeline:

-   **Data Loading & Exploration:** Getting to know our data.
-   **Preprocessing:** Cleaning and preparing the data for modeling (handling missing values, encoding, scaling).
-   **Exploratory Data Analysis (EDA):** Uncovering insights and patterns through visualizations.
-   **Model Selection & Training:** Experimenting with various models like Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting.
-   **Model Evaluation:** Assessing performance with metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
-   **Hyperparameter Tuning:** Fine-tuning models for optimal performance.

Don't forget to check out the `docs/` directory for supplementary materials:
-   `docs/ML-Final.pdf`: A concise project summary or presentation.
-   `docs/Rationale for Model Selection.docx`: In-depth explanations behind our model choices.

## 🧠 Technical Deep Dive

This section provides a closer look at the technical aspects of the project, particularly focusing on the Neural Network model development.

### Data Processing

Our journey began with a robust dataset of 41,188 rows and 21 columns. Key steps included:
-   **Initial Data Cleaning:** Removed 12 duplicate records. Handled 'unknown' missing values in categorical columns by replacing them with the mode. Notably, the 'default' column had a significant number of 'unknown' values (8,597).
-   **Data Transformation:**
    -   Binary target encoding: 'yes'/'no' transformed to 1/0.
    -   Numerical features (age, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed) were scaled using `StandardScaler`.
    -   Other numerical features (duration, campaign, pdays, previous) were scaled using `MinMaxScaler`.
    -   Categorical features (job, marital, default, housing, loan, contact, day_of_week, poutcome) were processed with One-Hot Encoding.
    -   Month data was cyclically encoded using sine and cosine transformations to preserve its periodic nature.
    -   Education levels were ordinally encoded, mapping from 0 (illiterate) to 6 (university degree).

### Feature Engineering

Beyond raw data, we engineered features to enhance model performance:
-   **Cyclic Features:** Month data was transformed using sine and cosine functions to capture its cyclical pattern.
-   **Education Level Mapping:** Education levels were mapped to an ordinal scale, reflecting their inherent hierarchy.

### Feature Selection

To optimize our model and reduce dimensionality, we carefully selected features based on:
-   **Correlation Analysis:** Identified and removed highly correlated features to mitigate multicollinearity.
-   **Feature Importance:** Utilized Random Forest to pinpoint the top features explaining 90% of the variance.

The final, refined feature set comprised 19 impactful features:
-   `age`
-   `nr.employed`
-   `campaign`
-   `education`
-   `housing_yes`
-   `marital_married`
-   `poutcome_success`
-   `loan_yes`
-   `cons.conf.idx`
-   `day_of_week_thu`
-   `day_of_week_tue`
-   `month_sin`
-   `day_of_week_mon`
-   `day_of_week_wed`
-   `job_technician`
-   `default_no`
-   `cons.price.idx`
-   `job_blue-collar`
-   `month_cos`

### Model Development

Our core predictive engine is a sophisticated **Neural Network**:

-   **Architecture:**
    -   **Input Layer:** 19 neurons (one for each selected feature).
    -   **Hidden Layer 1:** 128 neurons with ReLU activation, Batch Normalization, and Dropout (0.3).
    -   **Hidden Layer 2:** 64 neurons with ReLU activation, Batch Normalization, and Dropout (0.2).
    -   **Hidden Layer 3:** 32 neurons with ReLU activation, Batch Normalization, and Dropout (0.2).
    -   **Output Layer:** 1 neuron with Sigmoid activation for binary classification.

-   **Hyperparameter Tuning:** We meticulously tuned our model to find the optimal configuration:
    -   **Batch Size:** 32 (selected from [32, 64, 128])
    -   **Learning Rate:** 0.001 (selected from [0.001, 0.01])
    -   **Dropout Rate:** 0.2 (selected from [0.2, 0.3])
    -   **Optimizer:** Adam (selected from [Adam, RMSprop])

-   **Training Strategy:**
    -   **Early Stopping:** Implemented with `patience=5` to prevent overfitting.
    -   **ReduceLROnPlateau:** Dynamically adjusted the learning rate during training.
    -   Training typically ran for up to 150 epochs, often stopping earlier due to early stopping.

## 📈 Results and Analysis

The `notebooks/ML-Final.ipynb` notebook provides a detailed analysis of the Neural Network model's performance. Here's a snapshot of its capabilities:

### Performance Metrics
-   **Accuracy:** 0.8932
-   **ROC AUC Score:** 0.7609
-   **Precision:**
    -   Class 0 (No subscription): 0.90
    -   Class 1 (Subscription): 0.69
-   **Recall:**
    -   Class 0: 0.99
    -   Class 1: 0.16
-   **F1-score:**
    -   Class 0: 0.94
    -   Class 1: 0.26

### Cross-Validation Results
-   **5-fold CV Accuracy:** 0.8989 (±0.0027)

### Confusion Matrix
```
[[3601   35]
 [ 405   77]]
```

## 🤔 Rationale for Model Selection

The Artificial Neural Network (ANN) model emerged as our final choice due to its exceptional ability to handle the complexities of our dataset and its strong performance characteristics:

**1. Complex Feature Relationships:** The dataset's blend of categorical and numerical features, coupled with potentially non-linear relationships, made ANNs an ideal candidate. They excel at learning intricate patterns that simpler models might miss.

**2. Robust Performance:**
-   An impressive overall accuracy of 89.32% was achieved, which is substantial given the dataset's imbalance.
-   An AUC-ROC score of 0.7609 demonstrates good discriminative ability between subscribers and non-subscribers.
-   Consistent performance across different data splits was observed through cross-validation, with a low standard deviation (±0.0027), indicating model stability.

**3. Class Imbalance Handling:** Despite the significant class imbalance, the model maintained reasonable performance. While recall for the minority class (subscribers) is 16%, this is a common challenge. Crucially, the model successfully identified 77 potential subscribers out of 482, providing valuable leads for marketing efforts.

**4. Model Stability:** The integration of regularization techniques like Dropout and Batch Normalization effectively prevented overfitting. Learning rate scheduling via `ReduceLROnPlateau` further ensured stable convergence during training.

**5. Practical Applications:** Even modest improvements in identifying potential subscribers can yield significant business impact in a marketing context. The model's probability scores can be directly used to prioritize marketing efforts, making campaigns more efficient.

**6. Limitations:** It's important to acknowledge the low recall for the positive class (16%), meaning some potential subscribers might be missed. For real-world applications, adjusting the classification threshold might be necessary to increase sensitivity, potentially at the expense of some precision.

In essence, the ANN model strikes a commendable balance between accuracy and generalizability, aligning perfectly with our project's objective of identifying potential term deposit subscribers. While there's always room for refinement, this model provides a solid foundation for enhancing the bank's marketing strategy.

## 🚀 Conclusion & Future Work

The developed Neural Network model offers a reliable prediction system, empowering the bank's marketing team to more effectively identify potential term deposit subscribers. Its strength lies in its capacity to process diverse feature types and uncover complex relationships within the data.

To further elevate performance, particularly in detecting the positive class (subscribers), future explorations could include:

1.  **Advanced Imbalance Techniques:** Investigating methods like SMOTE or class weights.
2.  **Ensemble Methods:** Combining the ANN with other models for a more robust prediction.
3.  **Enhanced Feature Engineering:** Developing even more predictive variables.
4.  **Threshold Adjustment:** Fine-tuning the classification threshold to favor recall over precision, depending on business needs.

Despite these avenues for improvement, the current model already delivers valuable insights for optimizing marketing strategies and resource allocation.

## 👋 Contributing

We welcome contributions! If you have ideas for improvements, new features, or spot any bugs, please don't hesitate to open an issue or submit a pull request. Let's make this project even better together!

## 📜 License

This project is proudly licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as per the license terms.