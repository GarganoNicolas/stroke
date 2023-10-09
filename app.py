import gradio as gr
import pandas as pd
import pickle
import os

# Define params names
PARAMS_NAME = [
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "avg_glucose_level",
            "bmi",
            "smoking_status"
]

           
# Load model
with open("model1.pkl", "rb") as f:
    model = pickle.load(f)


import os

# Hacking my own protocol
os.chmod('saved_bins_bmi.pkl', 0o777)

with open('saved_bins_bmi.pkl', 'rb') as handle:
    saved_bins_bmi = pickle.load(handle)


def predict(*args):
    answer_dict = {}

    for i in range(len(PARAMS_NAME)):
        answer_dict[PARAMS_NAME[i]] = [args[i]]

    # Crear dataframe
    single_instance = pd.DataFrame.from_dict(answer_dict)


    single_instance["bmi"] = pd.cut(single_instance['bmi'],
                                     bins=saved_bins_bmi, 
                                     include_lowest=True)
    single_instance['bmi'] = single_instance['bmi'].cat.add_categories('null')

    single_instance_numbers = single_instance
    
    for columna in single_instance_numbers:
            # Verificar si el tipo de dato es "object"
            if single_instance_numbers[columna].dtype == 'object':
                # Obtener los valores únicos de la columna
                valores_unicos = single_instance_numbers[columna].unique()
                
                # Crear un diccionario de reemplazo
                diccionario_reemplazo = {valor: indice for indice, valor in enumerate(valores_unicos)}
                
                # Reemplazar los valores en la columna
                single_instance_numbers[columna] = single_instance_numbers[columna].map(diccionario_reemplazo)

    dataEnd_ohe = pd.get_dummies(single_instance_numbers).fillna(0)

    
    prediction = model.predict(dataEnd_ohe)


    # Cast numpy.int64 to just a int
    stroke = int(prediction[0])


    # Adaptación respuesta
    response = stroke
    if stroke == 1:
        response = "Keep rockin' babe!"
    if stroke == 0:
        response = "This brain will colapse in 3.. 2.. 1.. 🤯 "


    return response


with gr.Blocks() as demo:
    gr.Markdown(
        """
        #   Stroke Prevention 🤯
        """
    )

    with gr.Row():
        with gr.Column():

            gr.Markdown(
                """
                ## Insert your self data here please 🤓
                """
            )
            
            gender = gr.Radio(
                label='Gender',
                choices=['Male', 'Female'],
                value='Female',
            )

            age = gr.Slider(
                label='Age',
                minimum=35.0,
                maximum=82.0,
                step=1,
                randomize=True
            )

            hypertension = gr.Radio(
                label='Hypertension',
                choices=['No', 'Yes'],
                value='No',
            )

            heart_disease = gr.Radio(
                label='Heart Disease',
                choices=['Yes', 'No'],
                value='No',
            )

            ever_married = gr.Radio(
                label='Ever Married',
                choices=['Yes', 'No'],
                value='Yes',
            )

            work_type = gr.Radio(
                label='Work Type',
                choices=['Private', 'Self-employed', 'Govt-job'],
                value='Private',
            )

            Residence_type = gr.Radio(
                label='Residence Type',
                choices=['Urban', 'Rural'],
                value='Urban',
            )

            avg_glucose_level = gr.Slider(
                label='Avg Glucose Level',
                minimum=55.22,
                maximum=271.74,
                step=0.1,
                randomize=True
            )

            bmi = gr.Slider(
                label='Bmi',
                minimum=11.3,
                maximum=92.0,
                step=0.1,
                randomize=True
            )

            smoking_status = gr.Dropdown(
                label='Smoking Status',
                choices=['formerly smoked', 'never smoked', 'smokes', 'Unknown'],
                multiselect=False,
                value='never smoked',
            )         




        with gr.Column():

            gr.Markdown(
                """
                ## Look if your brain is in risk 🧠
                """
            )

            label = gr.Label(label="Brain status")
            predict_btn = gr.Button(value="Click me please!")
            predict_btn.click(
                predict,
                inputs=[
                    gender,
                    age,
                    hypertension,
                    heart_disease,
                    ever_married,
                    work_type,
                    Residence_type,
                    avg_glucose_level,
                    bmi,
                    smoking_status,
                ],
                outputs=[label],
                api_name="prediccion"
            )
            
            gr.Markdown(
                """
                ## <img src="https://media.giphy.com/media/ijb5ZE9zIQ2Nq/giphy.gif" alt="GIF">
                """
            )

    gr.Markdown(
        """
        <p style='text-align: center'>
            <a href='https://www.escueladedatosvivos.ai/cursos/bootcamp-de-data-science' 
                target='_blank'>Proyecto demo creado en el bootcamp de EDVAI 🤗
            </a>
        </p>
        <p style='text-align: center'>
            <a href='https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset' 
                target='_blank'>Data From IStroke Prediction Dataset update by Fede Soriano
            </a>
        </p>
        """
    )

demo.launch()
