
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Gaming Accessories Market Intelligence", layout="wide")

st.title("Gaming Accessories Data Intelligence Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("gaming_accessories_survey_synthetic_2000.csv")
    return df

df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Dataset Overview",
    "Descriptive Analytics",
    "Customer Segmentation (Clustering)",
    "Purchase Prediction (Classification)",
    "Budget Prediction (Regression)",
    "Association Rules (Product Bundles)",
    "Predict New Customers"
])

# ---------- DATASET OVERVIEW ----------
if page == "Dataset Overview":
    st.header("Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)

# ---------- DESCRIPTIVE ANALYTICS ----------
elif page == "Descriptive Analytics":
    st.header("Descriptive Analytics")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="Age_Group")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="Income")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x="Gaming_Platform")
    st.plotly_chart(fig, use_container_width=True)

# ---------- CLUSTERING ----------
elif page == "Customer Segmentation (Clustering)":

    st.header("Customer Segmentation")

    features = ["Gaming_Hours_Per_Week","Price_Sensitivity","Spending_Per_Purchase"]
    data = df[features].copy()

    le = LabelEncoder()
    for col in data.columns:
        data[col] = le.fit_transform(data[col])

    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    fig = px.scatter(df, x="Gaming_Hours_Per_Week", y="Spending_Per_Purchase", color="Cluster")
    st.plotly_chart(fig, use_container_width=True)

# ---------- CLASSIFICATION ----------
elif page == "Purchase Prediction (Classification)":
    st.header("Purchase Likelihood Prediction")

    data = df.copy()
    target = "Purchase_Likelihood"

    le = LabelEncoder()

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = le.fit_transform(data[col])

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test,y_pred,average="weighted")
    rec = recall_score(y_test,y_pred,average="weighted")
    f1 = f1_score(y_test,y_pred,average="weighted")

    st.write("Accuracy:",acc)
    st.write("Precision:",prec)
    st.write("Recall:",rec)
    st.write("F1 Score:",f1)

    try:
        roc = roc_auc_score(y_test,y_prob)
        st.write("ROC AUC:",roc)
    except:
        st.write("ROC AUC not available for multi-class")

    importance = model.feature_importances_
    feat_imp = pd.DataFrame({"Feature":X.columns,"Importance":importance})
    feat_imp = feat_imp.sort_values("Importance",ascending=False)

    fig = px.bar(feat_imp,x="Importance",y="Feature",orientation="h")
    st.plotly_chart(fig,use_container_width=True)

# ---------- REGRESSION ----------
elif page == "Budget Prediction (Regression)":
    st.header("Spending Power Prediction")

    data = df.copy()
    target = "Annual_Setup_Budget"

    le = LabelEncoder()

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = le.fit_transform(data[col])

    X = data.drop(columns=[target])
    y = data[target]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train,y_train)

    preds = model.predict(X_test)

    st.write("Regression model trained successfully.")

# ---------- ASSOCIATION RULES ----------
elif page == "Association Rules (Product Bundles)":

    st.header("Product Association Analysis")

    items = df[["Own_Keycaps","Own_Coiled_Cable","Own_Deskmat","Own_Cable_Sleeves","Own_Streaming_Gear"]]

    frequent = apriori(items, min_support=0.05, use_colnames=True)

    rules = association_rules(frequent, metric="confidence", min_threshold=0.3)

    st.write(rules[["antecedents","consequents","support","confidence","lift"]])

# ---------- NEW CUSTOMER PREDICTION ----------
elif page == "Predict New Customers":

    st.header("Upload New Customer Data")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:

        new_data = pd.read_csv(uploaded)

        st.write("Uploaded Data", new_data.head())

        data = df.copy()
        target = "Purchase_Likelihood"

        le = LabelEncoder()

        for col in data.columns:
            if data[col].dtype == "object":
                data[col] = le.fit_transform(data[col])

        X = data.drop(columns=[target])
        y = data[target]

        model = RandomForestClassifier()
        model.fit(X,y)

        for col in new_data.columns:
            if new_data[col].dtype == "object":
                new_data[col] = le.fit_transform(new_data[col])

        preds = model.predict(new_data)

        new_data["Predicted_Purchase_Likelihood"] = preds

        st.write(new_data)
