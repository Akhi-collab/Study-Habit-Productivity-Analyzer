# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- Load dataset ---
df = pd.read_csv("study.csv")

# --- Train model ---
X = df[['study_hours','breaks','mood']]
y = df['productivity_score']
model = LinearRegression()
model.fit(X, y)

# --- Streamlit App ---
st.set_page_config(
    page_title="Study Habit Productivity Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Study Habit Productivity Analyzer")
st.markdown("""
Predict your productivity based on your **study hours, breaks, and mood**.  
Adjust the sliders in the sidebar to see predictions and visualizations instantly.
""")

# --- Sidebar for inputs ---
st.sidebar.header("Input Your Study Data")
study_hours = st.sidebar.slider("Study Hours", 0.0, 10.0, 3.0, 0.1)
breaks = st.sidebar.slider("Number of Breaks", 0, 5, 1)
mood = st.sidebar.slider("Mood (1-5)", 1, 5, 3)

# --- Prediction ---
pred = model.predict(pd.DataFrame([[study_hours, breaks, mood]],
                                  columns=['study_hours','breaks','mood']))[0]

st.subheader("🔮 Predicted Productivity Score")
st.metric(label="Productivity Score", value=round(pred, 2))

# --- Layout for plots side by side ---
st.subheader("📈 Visualizations")
col1, col2, col3 = st.columns(3)

# Study Hours vs Productivity
with col1:
    fig, ax = plt.subplots()
    ax.scatter(df['study_hours'], df['productivity_score'], color='#1f77b4')
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Productivity Score")
    ax.set_title("Study Hours vs Productivity")
    st.pyplot(fig)

# Breaks vs Productivity
with col2:
    fig2, ax2 = plt.subplots()
    ax2.scatter(df['breaks'], df['productivity_score'], color='#2ca02c')
    ax2.set_xlabel("Breaks")
    ax2.set_ylabel("Productivity Score")
    ax2.set_title("Breaks vs Productivity")
    st.pyplot(fig2)

# Mood vs Productivity
with col3:
    fig3, ax3 = plt.subplots()
    ax3.scatter(df['mood'], df['productivity_score'], color='#9467bd')
    ax3.set_xlabel("Mood")
    ax3.set_ylabel("Productivity Score")
    ax3.set_title("Mood vs Productivity")
    st.pyplot(fig3)

# --- Instructions and insights ---
st.markdown("""
### How to use:
1. Adjust sliders in the **sidebar** for Study Hours, Breaks, and Mood.  
2. See the predicted productivity score instantly.  
3. Explore scatter plots to understand how each factor influences productivity.

### Tips:
- Higher study hours and better mood usually increase productivity.  
- Too many breaks may lower productivity.  
- Use this tool to plan your study schedule effectively!
""")

# Optional: Footer
st.markdown("---")
st.markdown("Created by **Akhila Reddy** | Powered by Python, Streamlit, Pandas, Matplotlib & Scikit-learn")