# Layton-Models: ML Classifiers for TCP/IP Flow Analysis  

This project explores the use of **machine learning classifiers** for TCP/IP flow analysis, following the **CRISP-DM** methodology. The models are trained using the **CIC-BCC-NRC-TabularioT-2024** dataset with the aim of being implemented in an **ML-based Network Intrusion Detection System (ML-NIDS)**.  

The trained models are designed to **analyze and classify IoT network flows**, providing an additional detection layer to **signature-based** and **flow-based IDS**.  

---

## Hypothesis  

The central hypothesis is whether it is possible to create reliable models for cyberattack analysis that can be deployed in real environments.  

To test this, a dataset composed of multiple sources and environments was used in order to verify if models can learn from heterogeneous data and generalize attack patterns to real-world scenarios.  

---

## Methodology  

The project followed the **CRISP-DM** methodology, commonly used in data science projects.  
Data was preprocessed and models were trained iteratively, with each step followed by analysis and key findings.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/8fa9ac5d-5ae6-47e5-88f9-0c620e136602" width="600" alt="CRISP-DM diagram" />
</p>  

---

## Key Findings  

- While most ML-NIDS research shows that it is possible to train highly accurate ML-based IDS, this is often not true in **production environments**.  
- When trained on one dataset and evaluated on another (**cross-dataset evaluation**), models do not generalize properly.  
  - Instead of learning attack patterns, they overfit to the characteristics of the lab environment.  
- However, training classifiers on **multiple datasets** seems promising.  
  - Results may not match the "astounding" accuracies often reported in the literature, but they are **high enough to be practical and fair**.  

---

## Models & Results  

Two final models were trained:  

- **Binary classifier**  
  - Accuracy: **91.65%**  
  - Confusion matrix:  
    <p align="center">
      <img src="https://github.com/user-attachments/assets/366dc6b6-ee8a-488e-8246-251257ee4444" width="600" alt="Confusion matrix binary classifier" />
    </p>  

- **Multiclass classifier**  
  - Accuracy: **96%**  
  - Confusion matrix:  
    <p align="center">
      <img src="https://github.com/user-attachments/assets/6456f274-c43c-4a3e-927b-e993d39cd07e" width="600" alt="Confusion matrix multiclass classifier" />
    </p>  

---

## References & Acknowledgments  

- **Dataset**: Tinshu Sasi, Arash Habibi Lashkari, Rongxing Lu, Pulei Xiong, Shahrear Iqbal,  
  *An Efficient Self Attention-Based 1D-CNN-LSTM Network for IoT Attack Detection and Identification Using Network Traffic*, Journal of Information and Intelligence, 2024.  

- **CICFlowMeter**: Arash Habibi Lashkari, Gerard Draper-Gil, Mohammad Saiful Islam Mamun and Ali A. Ghorbani,  
  *Characterization of Tor Traffic Using Time Based Features*, Proceedings of the 3rd International Conference on Information System Security and Privacy, SCITEPRESS, Porto, Portugal, 2017.  
