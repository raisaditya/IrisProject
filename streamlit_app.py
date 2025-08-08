import streamlit as st
import pandas as pd
import joblib

# how to run streamlit app locally
# 1 open terminal and run 'streamlit run streamlit_app.py'


st.title('Iris Classifier') # membuat judul aplikasi

st.write('This is a simple Iris Classifier app') # berfungsi untuk menampilkan text

# inference function
# model = joblib.load('model.joblib')


def get_prediction (data:pd.DataFrame):
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    return pred, pred_proba


# user input
left, right = st.columns(2, gap = 'medium', border = True)

# -- Sepal Input
left.subheader ('Sepal')
sepal_length = left.slider('Sepal Length', min_value = 4.3, max_value = 10.0, value = 5.4, step = 0.1) # harus float agar kebaca
sepal_width = left.slider('Sepal Width', min_value = 4.3, max_value = 10.0, value = 5.4, step = 0.1)

# -- Petal Input
right.subheader ('Petal')
petal_length = right.slider('Petal Length', min_value = 4.3, max_value = 10.0, value = 5.4, step = 0.1)
petal_width = right.slider('Petal Width', min_value = 4.3, max_value = 10.0, value = 5.4, step = 0.1)

# show input value
# st.dataframe() berfungsi untuk menampilkan dataframe
# use_container_width mengatur lebar dari dataframe
# True lebar df sesuai dg lebar container

data = pd.DataFrame ({'sepal length (cm)' : [sepal_length],
                      'sepal width (cm)' : [sepal_width],
                      'petal length (cm)' : [petal_length],
                      'petal width (cm)' : [petal_width]})

st.dataframe(data, use_container_width = True)

# prediction button
button = st.button ('Predict', use_container_width = True)

if button:
    st.write ('Prediksi Berhasil !')
    pred, pred_proba = get_prediction(data)

    label_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

    label_pred = label_map[pred[0]]
    label_proba = pred_proba[0][pred[0]]
    output = f'Iris diklasifikasikan sebagai {label_proba: .0%} {label_pred}'
    st.write(output)