# Bank Term Deposit Prediction

This project focuses on predicting whether a client will subscribe to a term deposit, based on the "Bank Marketing" dataset. The goal is to build a machine learning model that can help banking institutions identify potential customers for term deposits, optimizing their marketing campaigns and resource allocation.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Analysis](#results-and-analysis)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The banking sector frequently uses targeted marketing campaigns to acquire new customers for various products, including term deposits. This project leverages historical campaign data to develop a predictive model. By analyzing customer demographics, past campaign interactions, and other relevant attributes, the model aims to predict the likelihood of a customer subscribing to a term deposit. This can significantly improve the efficiency and effectiveness of marketing efforts.

## Dataset

The dataset used in this project is the "Bank Marketing" dataset, publicly available from the UCI Machine Learning Repository. It contains information about direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable `y`).

The dataset includes:
- **Client data:** age, job, marital status, education, default, housing, loan.
- **Last contact data:** contact communication type, month, day of week, duration.
- **Other attributes:** campaign, pdays, previous, poutcome.
- **Social and economic context attributes:** euribor3m, job_variation_rate, consumer_price_index, consumer_confidence_index, euribor3m, number_employed.
- **Output variable (desired target):** `y` - has the client subscribed a term deposit? (binary: 'yes', 'no').

The raw data files are located in the `data/` directory:
- `bank-additional-full.csv`: Full dataset with all examples and attributes.
- `bank-additional.csv`: Smaller version of the dataset.
- `bank-additional-names.txt`: Description of the dataset attributes.

## Project Structure

The project is organized into the following directories:

```
.
├── data/
│   ├── bank-additional-full.csv
│   ├── bank-additional-names.txt
│   └── bank-additional.csv
├── notebooks/
│   └── ML-Final.ipynb
├── docs/
│   ├── ML-Final.pdf
│   └── Rationale for Model Selection.docx
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

- `data/`: Contains the raw and processed datasets.
- `notebooks/`: Contains Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- `docs/`: Contains project documentation, reports, and presentations.
- `.gitignore`: Specifies intentionally untracked files to ignore.
- `LICENSE`: The license under which this project is distributed.
- `README.md`: This file, providing an overview of the project.
- `requirements.txt`: Lists the Python dependencies required to run the project.

## Installation

To set up the project environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/Bank-Term-Deposit-Prediction.git
    cd Bank-Term-Deposit-Prediction
    ```
    *(Note: You will need to create the repository on GitHub first and replace `YOUR_USERNAME` with your GitHub username.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

After setting up the environment, you can explore the project:

1.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook notebooks/ML-Final.ipynb
    ```
    This notebook contains the complete workflow, including:
    - Data loading and initial exploration.
    - Data preprocessing (handling missing values, encoding categorical features, scaling numerical features).
    - Exploratory Data Analysis (EDA) and visualizations.
    - Model selection and training (e.g., Logistic Regression, Decision Trees, Random Forests, Gradient Boosting).
    - Model evaluation using appropriate metrics (accuracy, precision, recall, F1-score, ROC-AUC).
    - Hyperparameter tuning.

2.  **Review Documentation:**
    - Check `docs/ML-Final.pdf` for a summary of the project or presentation.
    - Read `docs/Rationale for Model Selection.docx` for detailed explanations of model choices.

## Results and Analysis

The `notebooks/ML-Final.ipynb` notebook provides a detailed analysis of the model's performance, including:
- Key findings from EDA.
- Performance metrics of different models.
- Feature importance analysis.
- Conclusions and potential next steps.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).