import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from mlxtend.frequent_patterns import apriori, association_rules

# Set Page Config
st.set_page_config(page_title="Anshul's Gaming Analytics", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('gaming_market_india.csv')
        return df
    except FileNotFoundError:
        st.error("CSV not found. Please run the data generator first!")
        return None

df = load_data()

st.title("🎮 Gaming Market Entry: Data-Driven Founder's Dashboard")
st.markdown("---")

if df is not None:
    # --- SIDEBAR: NEW CUSTOMER PREDICTION ---
    st.sidebar.header("Target New Customers")
    with st.sidebar.expander("Predict Individual Inclination"):
        s_age = st.number_input("Age", 10, 60, 25)
        s_income = st.number_input("Monthly Income", 0, 500000, 50000)
        s_friction = st.slider("Import Friction (1-10)", 1, 10, 5)
        if st.button("Predict Interest"):
            # Simple logic-based prediction for the UI
            score = (s_income/250000)*0.4 + (s_friction/10)*0.5
            res = "HIGH" if score > 0.5 else "LOW"
            st.success(f"Customer Potential: {res}")

    # --- 1. DESCRIPTIVE ANALYTICS ---
    st.header("1. Descriptive Analytics: Market Pulse")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.pie(df, names='Persona_Label', title="Customer Segment Distribution", 
                     color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        fig2 = px.histogram(df, x='Monthly_Income', color='City_Tier', barmode='group',
                           title="Income Distribution by City Tier", color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig2, use_container_width=True)

    # --- 2. DIAGNOSTIC ANALYTICS ---
    st.header("2. Diagnostic Analytics: The Why")
    col3, col4 = st.columns(2)
    
    with col3:
        # Association Rule Mining (Simplified for display)
        basket = df[['Int_Devanagari_Keycaps', 'Int_Coiled_Cables', 'Int_Desk_Mats', 'Int_Mobile_Grips']]
        frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        st.subheader("Product Associations (Market Basket)")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))

    with col4:
        fig3 = px.scatter(df, x="Monthly_Income", y="Import_Friction_Score", color="Target_Will_Purchase",
                         title="Correlation: Income vs. Import Pain", color_discrete_map={'Yes':'#00CC96', 'No':'#EF553B'})
        st.plotly_chart(fig3, use_container_width=True)

    # --- 3. PREDICTIVE ANALYTICS ---
    st.header("3. Predictive Analytics: ML Performance")
    
    # Classification: Predicting Purchase
    X = df[['Age', 'Monthly_Income', 'Import_Friction_Score']]
    y = df['Target_Will_Purchase'].apply(lambda x: 1 if x=='Yes' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:, 1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    m2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
    m3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
    m4.metric("F1-Score", f"{f1_score(y_test, y_pred):.2f}")

    # Feature Importance Plot
    feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
    fig4 = px.bar(feat_importances, title="Driver Identification (Classification)", 
                 labels={'value':'Importance', 'index':'Feature'}, color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig4, use_container_width=True)

    # --- 4. PRESCRIPTIVE ANALYTICS ---
    st.header("4. Prescriptive: Strategic Grouping")
    
    # Clustering for segmenting discounts
    X_cluster = df[['Age', 'Monthly_Income']]
    kmeans = KMeans(n_clusters=3, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster)
    
    fig5 = px.scatter(df, x='Age', y='Monthly_Income', color='Cluster', 
                     title="Clustering for Customized Discounting",
                     color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig5, use_container_width=True)
    
    st.info("**Founder's Action Plan:** Target the high-income clusters with 'Premium Early Access' and the younger clusters with 'Student Bundle Discounts'.")
