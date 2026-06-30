import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import streamlit as st

st.title("Health insurance project")
st.header("know your charge")


df = pd.read_csv("Health insurance project.csv")
df = df.fillna(df.mean())


X = df[["age" , "bmi", "children" , "smoker"]]
Y = df["charges"]

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=.2,random_state=42)


model = LinearRegression()
model.fit(x_train, y_train)


pred = model.predict(x_test)
acc = r2_score(y_test, pred)
print("-" * 40)
print(f"🎯 Insurance Model Accuracy: {acc * 100:.2f}%")


smokers_avg = df[df["smoker"] == 1]["charges"].mean()
not_smokers_avg = df[df["smoker"] == 0]["charges"].mean()
print(f"🚬 Avg charges for smokers: {smokers_avg:.2f} $")
print(f"🍏 Avg charges for non-smokers: {not_smokers_avg:.2f} $")
print("-" * 40)


st.subheader("📊 أدخل بيانات العميل لحساب التكلفة:")


age = st.number_input("السن (Age):", min_value=18, max_value=100, value=30)
bmi = st.number_input("مؤشر كتلة الجسم (BMI):", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("عدد الأطفال (Children):", min_value=0, max_value=10, value=0)


is_smoker = st.checkbox("هل العميل مدخن؟ (Smoker)")

smoker_val = 1 if is_smoker else 0


if st.button("Calculate Insurance Charges"):

    user_data = pd.DataFrame([[age, bmi, children, smoker_val]], columns=["age", "bmi", "children", "smoker"])
    

    prediction = model.predict(user_data)
    

    st.success(f"💵 تكلفة التأمين الصحي السنوية المتوقعة: ${prediction[0]:.2f}")
