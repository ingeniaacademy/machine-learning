import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import io

# Load the Keras model for tumor detection
@st.cache(allow_output_mutation=True)
def load_tumor_model():
    model_url = 'https://resnet-model-10.s3.ap-southeast-1.amazonaws.com/2019-8-6_resnet50.h5'
    response = requests.get(model_url)
    
    # Save model to local file
    with open('tumor_model.h5', 'wb') as f:
        f.write(response.content)
    
    # Load model from local file
    model = tf.keras.models.load_model('tumor_model.h5')
    return model

# Define the labels for the tumor classes
tumor_labels = ['YES', 'NO']

# Load the generative AI model for generating MRI images
@st.cache(allow_output_mutation=True)
def load_generative_model():
    generative_model = tf.keras.models.load_model('generative_model.h5')
    return generative_model

# Define a function to preprocess the image for tumor detection
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Define a function to generate an MRI image using generative AI
def generate_mri_image(generative_model):
    # Generate a random noise vector
    noise = tf.random.normal([1, 100])
    # Generate an MRI image from the noise vector
    generated_image = generative_model.predict(noise)
    # Rescale the pixel values to the range [0, 255]
    generated_image = (generated_image * 127.5 + 127.5).astype(np.uint8)
    # Convert the generated image array to a PIL image
    generated_image = Image.fromarray(generated_image[0])
    return generated_image

# Define the Streamlit app
def main():
    st.title('MRI Toolkit')

    # Sidebar menu to select the tool
    tool = st.sidebar.selectbox('Select Tool', ['Tumor Detection', 'Generate MRI Image'])

    if tool == 'Tumor Detection':
        st.subheader('Tumor Detection')
        uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded MRI Image', use_column_width=True)
            st.write("")

            # Load the tumor detection model
            tumor_model = load_tumor_model()

            # Preprocess the image for tumor detection
            preprocessed_image = preprocess_image(image)

            # Make prediction
            prediction = tumor_model.predict(preprocessed_image)

            # Get the predicted tumor class label
            predicted_class = tumor_labels[np.argmax(prediction)]

            st.write("Prediction:", predicted_class)

    elif tool == 'Generate MRI Image':
        st.subheader('Generate MRI Image')
        st.write("Click the button below to generate a new MRI image:")

        # Load the generative AI model
        generative_model = load_generative_model()

        if st.button('Generate'):
            generated_image = generate_mri_image(generative_model)
            st.image(generated_image, caption='Generated MRI Image', use_column_width=True)

if __name__ == "__main__":
    main()
