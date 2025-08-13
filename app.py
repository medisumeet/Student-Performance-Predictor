import streamlit as st
import numpy as np
import pandas as pd
from gdr import GDRegressor
from sklearn.model_selection import train_test_split


@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\HP\project\student_performance_linear_ready.csv")
    X = df[["Study_Hours_per_Week", "Homework_Completion_Rate", "Previous_Semester_GPA"]]
    y = df["Final_Score"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


@st.cache_resource
def train_model(X_train, y_train):
    gdr = GDRegressor()
    gdr.fit(X_train, y_train)
    return gdr

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)

    st.title("Student Performance Predictor")

    if "page" not in st.session_state:
        st.session_state.page = "welcome"

    if st.session_state.page == "welcome":
        st.markdown("### Do you want to predict the student's final score?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes"):
                st.session_state.page = "input"
        with col2:
            if st.button("No"):
                st.write("Okay! Come back anytime to predict.")

    elif st.session_state.page == "input":
        st.markdown("### Enter the student details")

        study_hours = st.number_input(
            "Study Hours per Week", min_value=0.0, max_value=50.0, value=0.0, step=0.1, format="%.1f"
        )
        homework_completion = st.number_input(
            "Homework Completion Rate (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f"
        )
        prev_gpa = st.number_input(
            "Previous Semester GPA", min_value=0.0, max_value=10.0, value=0.0, step=0.01, format="%.2f"
        )

        if st.button("Predict"):
            st.session_state.input_data = [study_hours, homework_completion, prev_gpa]
            st.session_state.page = "output"

        if st.button("Back"):
            st.session_state.page = "welcome"

    elif st.session_state.page == "output":
        input_data = np.array(st.session_state.input_data).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        prediction = max(0, min(100, prediction))

        st.markdown("### Prediction Result")
        st.write(f"Predicted Final Score: **{prediction:.2f}**")

        if st.button("Predict Again"):
            st.session_state.page = "input"

        if st.button("Exit"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.page = "welcome"

if __name__ == "__main__":
    main()
