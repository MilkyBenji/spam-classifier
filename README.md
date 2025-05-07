# **Interactive Spam Classifier**  

**My interactive Spam Classifier** is a machine learning project that utilizes **Naïve Bayes classification** to detect whether a message is spam or not. The model is deployed using **Gradio**, allowing users to enter text and receive real-time classifications.

---

## **Table of Contents**  

1. [Project Overview](#project-overview)  
2. [Video Demo](#video-demo)  
3. [Motivation and Purpose](#motivation-and-purpose)  
4. [Problem Statement and Objectives](#problem-statement-and-objectives)  
5. [Dataset Description](#dataset-description)  
6. [Model Architecture](#model-architecture)  
7. [Installation and Usage](#installation-and-usage)  
8. [Evaluation and Results](#evaluation-and-results)  
9. [Contributing](#contributing)  
10. [License](#license)  

---

## **Project Overview**  

### **Introduction**  
This project implements a **Naïve Bayes-based spam detection model** with an **interactive web interface** using **Gradio**. The classifier predicts whether an input message is **Spam** or **Not Spam** based on textual features.

---

## **Video Demo**  

https://youtu.be/FfFg3h3NezA

---

## **Motivation and Purpose**  
Spam messages are a persistent issue in digital communication, affecting **email services, SMS platforms, and chat applications**. This project aims to:  

- Build an interactive **spam classification system**.  
- Explore the effectiveness of **Naïve Bayes** for text classification.  
- Provide an easy-to-use **Gradio interface** for real-time spam detection.  

---

## **Problem Statement and Objectives**  

### **Problem Statement:**  
Identifying spam messages manually is inefficient and time-consuming. The objective of this project is to develop an **automated spam classifier** that can **accurately detect spam messages** using machine learning.

### **Objectives:**  
✔️ Train a **Naïve Bayes classifier** to classify text as **Spam** or **Not Spam**.  
✔️ Use **TF-IDF vectorization** to extract key features from text.  
✔️ Deploy the model using **Gradio** for real-time user interaction.  
✔️ Ensure high accuracy and robustness of the classifier.  

---

## **Dataset Description**  
The dataset consists of labeled messages used to train the spam classifier:

| Feature | Description |  
|---------|------------|  
| **Message** | The text content of the message |  
| **Label** | Spam (1) or Not Spam (0) |  

📌 **Data Preprocessing Steps:**  
- **Text Cleaning:** Removal of stopwords, punctuation, and special characters.  
- **Tokenization:** Breaking messages into individual words.  
- **TF-IDF Vectorization:** Converting text into numerical form for model training.  

---

## **Model Architecture**  
The **Naïve Bayes Classifier** is used for spam classification, following these steps:

1. **Text Preprocessing** – Cleaning and transforming message text.  
2. **Feature Extraction** – Converting text into numerical features.  
3. **Training the Model** – Using **Multinomial Naïve Bayes** for classification.  
4. **Deploying with Gradio** – Creating an interactive user interface.  

📌 **Hyperparameters Used:**  
- **Model**: Multinomial Naïve Bayes  
- **Feature Extraction**: TF-IDF Vectorization  
- **Stopwords Removal**: Applied for better classification  

---

## **Installation and Usage**  

### **🔧 Prerequisites**  
Install the necessary dependencies before running the project:  

```bash
pip install numpy pandas scikit-learn gradio
```

### **📌 Running the Model**  

#### **Clone the Repository:**  
```bash
git clone https://github.com/MilkyBenji/SpamClassifier.git
cd SpamClassifier
```

#### **Run the Application:**  
```bash
python app.py
```

This will launch the **Gradio interface**, allowing users to enter messages and receive predictions.

---

## **Evaluation and Results**  

### **📊 Model Performance**  
The **Naïve Bayes model** was evaluated using classification metrics:

- **Accuracy:** Measures overall correctness.  
- **Precision & Recall:** Determines effectiveness in detecting spam.  
- **Confusion Matrix:** Provides insight into classification performance.

📌 **Visualization:**  
- A **classification report** details precision, recall, and F1-score.  
- **Sample predictions** demonstrate the model's accuracy on unseen text.  

### **📈 Sample Prediction**  

Given a sample input:
```python
sample_message = "Congratulations! You have won a free iPhone. Click here to claim."
```

The model predicts:
```bash
Predicted Label: Spam
```

---

## **Contributing**  

If you'd like to contribute to **Interactive Spam Classifier**, feel free to **fork the repository** and submit a **pull request**. Contributions are always welcome!  

### **Guidelines:**  
✔️ **Follow Best Practices**: Ensure the code is clean and well-documented.  
✔️ **Testing**: Validate model performance before submitting any changes.  
✔️ **Feature Additions**: If suggesting enhancements, provide a detailed explanation.  

---

## **License**  

This project is licensed under the **MIT License** – see the `LICENSE` file for details.  

---

## **📌 Summary**  
🚀 This project applies **Naïve Bayes classification** to detect spam messages. With **Gradio's interactive UI**, users can test the model in real time, making it an efficient tool for spam detection.

