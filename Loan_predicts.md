## Table of Contents

**1. Exploratory Data Analysis (EDA)**
  <details><summary> 1.1 Target Variable Overview (Loan_Status) </summary>
```python
tra_df.set_index(tra_df.Loan_ID, inplace = True)
tra_df.drop('Loan_ID', axis=1, inplace=True)
tra_df.head() 
```
<img width="1062" height="193" alt="image" src="https://github.com/user-attachments/assets/2f20f00a-7339-4cf9-83fd-00f9590d64da" />
 

</details>
  <details><summary> 1.2 Impact Analysis of Demographic and Financial Variables) </summary>
    
```python
tra_df['Loan_Status'].value_counts(normalize=True) 
```

<img width="257" height="141" alt="image" src="https://github.com/user-attachments/assets/c5cee9df-8b23-4c71-90c5-06d088b4fee3" />

'*Gần* 69% các khoản vay sẽ được chấp nhận'

```python
cols = ['Gender','Married','Dependents','Education','Self_Employed','Credit_History']

n_rows = 2
n_cols = 3
fix, ax = plt.subplots(n_rows,n_cols,figsize=(n_cols*3.5,n_rows*3.5))

for r in range (0,n_rows):
    for c in range (0,n_cols):
        i = r*n_cols + c
        if i < len(cols):
            ax_i = ax[r,c]
            sns.countplot(data=tra_df,x = cols[i], hue = 'Loan_Status', ax=ax_i, palette= 'Purples')
plt.tight_layout() 
```
<img width="943" height="571" alt="image" src="https://github.com/user-attachments/assets/570648e5-64b6-439b-9433-fa29637f850a" />

  - Fig1: Người vay là nam giới nhiều hơn hẳn nữ giới
  - Fig2: Tỷ lệ chấp nhận vay của người đã kết hôn cao hơn người chưa kết hôn
  - Fig3: Người chưa có con cái có nhu cầu đi vay nhiều nhất
  - Fig4: Những người đã tốt nghiệp có xu hương vay nhiều hơn
  - Fig5: Những người làm thuê vay nhiều hơn
  - Fig6: Những người có lịch sử tín dụng xấu không được chấp nhận khoản vay



</details>

**2. Data Preprocessing**

  <details><summary> 2.1 Missing Values Imputation </summary>
    
```python
tra_df.isnull().sum() 
```
<img width="253" height="385" alt="image" src="https://github.com/user-attachments/assets/a95ee4fd-e7ec-46cc-8d65-c7c60c758050" />

</details>

  <details><summary> 2.2 Outlier Treatment (Capping / Winsorizing) </summary>
    
```python
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(y=tra_df[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel('')
plt.tight_layout()
plt.show() 
```
<img width="1261" height="403" alt="image" src="https://github.com/user-attachments/assets/d45fa605-c36f-4258-baf2-77f96a669f58" />


</details>
  <details><summary> 2.3 Categorical Feature Encoding & Dummy Variables </summary>

<img width="1518" height="189" alt="image" src="https://github.com/user-attachments/assets/b25fce21-f438-42bd-8eaf-fcd60b8b3cd4" />


</details>

**3. Model Building and Training**

  <details><summary> 3.1 Train/Test Data Split </summary>
    
```python
X = tra_df.drop(['Loan_Status'], axis=1)
y = tra_df['Loan_Status']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state = 1) 
```

</details>

  <details><summary> 3.2 Training Classification Algorithms (Logistic Regression, Decision Tree, Random Forest) </summary>
    
```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=1),
    "Random Forest": RandomForestClassifier(random_state=1)
}
#Huấn luyện và dự đoán
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred)) 
```

</details>

  <details><summary> 3.3 Baseline Model Performance Evaluation (Accuracy, Precision, Recall, F1-Score) </summary>

<img width="416" height="476" alt="image" src="https://github.com/user-attachments/assets/551fcecf-10fc-43ea-b7fe-0403548f04c0" />

</details>

**4. Optimization and Imbalanced Data Handling (Advanced Techniques)**
 <details><summary> 4.2 Applying SMOTE Algorithm for Data Balancing </summary>
   
```python
sm = SMOTE(random_state=1)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

lr_smote = LogisticRegression(max_iter=1000)
lr_smote.fit(X_train_res, y_train_res)

y_pred_smote = lr_smote.predict(X_test)

print("Tỷ lệ các lớp sau khi SMOTE:", pd.Series(y_train_res).value_counts())
print(classification_report(y_test, y_pred_smote)) 
```
<img width="395" height="181" alt="image" src="https://github.com/user-attachments/assets/1a554821-cd04-43b2-8c6b-1f7ab49b142b" />

</details>

  <details><summary> 4.2 Post-Optimization Performance Improvement Evaluation </summary>
   
```python
cm_smote = confusion_matrix(y_test, y_pred_smote)
plt.figure(figsize=(6,4))
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Confusion Matrix sau khi SMOTE')
plt.show() 
```
<img width="483" height="326" alt="image" src="https://github.com/user-attachments/assets/0441c52c-510b-45ab-ab09-34adb9c3f28f" />

</details>

**5. Deployment and Prediction**
<details><summary> Running Predictive Models and Extracting Results (DataFrame Output) </summary>
   
```python
# Dự báo kết quả (0 hoặc 1)
predictions = lr_smote.predict(X_test_final)

# Tạo dataframe kết quả
final_output = pd.DataFrame({
    'Loan_ID': te_df.index,
    'Loan_Status_Predicted': predictions
})

# Chuyển ngược lại 1 -> 'Y', 0 -> 'N' cho dễ đọc
final_output['Loan_Status_Predicted'] = final_output['Loan_Status_Predicted'].map({1: 'Y', 0: 'N'})
print(final_output)
sns.countplot(data=final_output, x='Loan_Status_Predicted')
plt.show() 
```
<img width="512" height="569" alt="image" src="https://github.com/user-attachments/assets/a459131c-7fda-4eee-8f92-a4e09e353770" />

</details>

