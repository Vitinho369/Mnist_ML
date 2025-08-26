from joblib import load
import streamlit as st
from streamlit_mnist_canvas import st_mnist_canvas

modelo_sgd= load('models/modelo_sgd.pkl')

st.markdown('<div class="drawImage">', unsafe_allow_html=True)
st.subheader("Desenhe um número por favor:")
result = st_mnist_canvas()
st.markdown('</div>', unsafe_allow_html=True)

if result.is_submitted:
    img = result.resized_grayscale_array

    img_reshaped = img.reshape(1, 784)
    result = modelo_sgd.predict(img_reshaped)


    if result[0]:
        st.write("De acordo com a IA é um número 5")
    else:
        st.write("De acordo com a IA não é um número 5")
    st.markdown('</div>', unsafe_allow_html=True)

st.caption('IA pode cometer erros. Considere verificar informações importantes')
