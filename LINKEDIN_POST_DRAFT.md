# LinkedIn post draft — Will This Visitor Buy?

E-commerce teams see many browsing sessions that do not end in a purchase. I explored whether historical session behavior can help prioritize optional customer assistance for further testing.

For my NextGen Institute Machine Learning project, I built a purchase-intent classification workflow and Streamlit prototype using the UCI Online Shoppers Purchasing Intention dataset: **12,330 sessions** with an imbalanced purchase target. I used Python, pandas, scikit-learn, joblib and Streamlit to compare Logistic Regression, Random Forest and Extra Trees against a majority-class baseline.

The Random Forest had the highest training cross-validation F1 (**0.4176**) and was selected. On the held-out test set, it achieved **0.7548 ROC-AUC**, **0.3425 PR-AUC**, **0.4764 recall** and **0.3688 F1** at threshold 0.50. Precision was **0.3008**, with 423 false positives and 200 false negatives.

The main lesson was that accuracy would hide how poorly a majority-class model detects purchases. The results also reminded me that a historical session score does not prove a coupon or chat causes a sale. Timestamped pre-intervention data and controlled experiments would be needed before using such a policy.

Try the app: [Will This Visitor Buy?](https://ecommerce-purchase-intent-prediction-kmstr57kdkbvvtffwkrtzd.streamlit.app/)  
Project code: [GitHub repository](https://github.com/Safaa-cloud/ecommerce-purchase-intent-prediction)


#MachineLearning #DataScience #Ecommerce #Python #ScikitLearn #Streamlit #NextGenInstitute
