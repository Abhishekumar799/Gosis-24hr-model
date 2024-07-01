import numpy as np
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
from PIL import Image

model = load_model('Model/lightgbm_Gosis_model')

cat_map = {
    "No": 0,
    "Yes": 1,
    "Not Available": np.nan,
    "Male": 1,
    "Female": 0
    
}


def predict(model, input_df):
    # model.memory = "Data/"
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['prediction_label'][0]
   # confidence = predictions_df['prediction_score'][0]
    return predictions


def get_data():
    # data = pd.read_csv("Data/peerj-08-10337-s001.csv")
    data = pd.read_csv("Data/gosis-1-24hr.csv")
    data.columns = list(map(str.strip, list(data.columns)))
    data = data[['bmi', 'hospital_los_days', 'icu_los_days', 'albumin_apache', 'apache_2_diagnosis', 'apache_3j_diagnosis', 'apache_post_operative', 'arf_apache', 'fio2_apache', 'map_apache', 'ph_apache', 'temp_apache', 'ventilated_apache', 'wbc_apache', 'd1_heartrate_max', 'd1_heartrate_min', 'd1_sysbp_invasive_min', 'h1_heartrate_max', 'h1_heartrate_min', 'd1_arterial_ph_max', 'd1_arterial_ph_min', 'apache_3j_score', 'apsiii']]
    return data


def main():
    data = get_data()
    # image2 = Image.open('Images/icu.png')
    # st.sidebar.info('This app is created to predict a particular patient need ICU treatment or no. [DeepAarogya]] - Version 2')
    # st.sidebar.image(image2)
    st.title("ICU Mortality 24 hr")

    st.sidebar.title("Check Analysis:")

   

    gender_M = st.selectbox('Male:', ["Not Available", "Yes", "No"])
    gender_F = st.selectbox('Female:', ["Not Available", "Yes", "No"])  

    if st.checkbox("Do you have patient BMI?", False):
        bmi = st.number_input('BMI:', min_value=data.describe()["bmi"].loc["min"],
                                    max_value=data.describe()["bmi"].loc["max"],
                                    value=data.describe()["bmi"].loc["50%"])
    else:
        bmi = np.nan


    if st.checkbox("Do you have patient hospital_los_days?", False):
        hospital_los_days = st.number_input('Hospital LOS Days:', 
                                        min_value=data.describe()["hospital_los_days"].loc["min"], 
                                        max_value=data.describe()["hospital_los_days"].loc["max"], 
                                        value=data.describe()["hospital_los_days"].loc["50%"])
    else:
        hospital_los_days = np.nan
   


    if st.checkbox("Do you have patient icu_los_days?", False):
        icu_los_days = st.number_input('ICU LOS Days:', 
                                    min_value=data.describe()["icu_los_days"].loc["min"], 
                                    max_value=data.describe()["icu_los_days"].loc["max"], 
                                    value=data.describe()["icu_los_days"].loc["50%"])
    else:
        icu_los_days = np.nan

    if st.checkbox("Do you have patient albumin_apache?", False):
        albumin_apache = st.number_input('Albumin APACHE:', 
                                        min_value=data.describe()["albumin_apache"].loc["min"], 
                                        max_value=data.describe()["albumin_apache"].loc["max"], 
                                        value=data.describe()["albumin_apache"].loc["50%"])
    else:
        albumin_apache = np.nan

    if st.checkbox("Do you have patient apache_2_diagnosis?", False):
        apache_2_diagnosis = st.number_input('APACHE 2 Diagnosis:', 
                                            min_value=data.describe()["apache_2_diagnosis"].loc["min"], 
                                            max_value=data.describe()["apache_2_diagnosis"].loc["max"], 
                                            value=data.describe()["apache_2_diagnosis"].loc["50%"])
    else:
        apache_2_diagnosis = np.nan

    if st.checkbox("Do you have patient apache_3j_diagnosis?", False):
        apache_3j_diagnosis = st.number_input('APACHE 3J Diagnosis:', 
                                            min_value=data.describe()["apache_3j_diagnosis"].loc["min"], 
                                            max_value=data.describe()["apache_3j_diagnosis"].loc["max"], 
                                            value=data.describe()["apache_3j_diagnosis"].loc["50%"])
    else:
        apache_3j_diagnosis = np.nan

    if st.checkbox("Do you have patient apache_post_operative?", False):
        apache_post_operative = st.number_input('APACHE Post Operative:', 
                                                min_value=data.describe()["apache_post_operative"].loc["min"], 
                                                max_value=data.describe()["apache_post_operative"].loc["max"], 
                                                value=data.describe()["apache_post_operative"].loc["50%"])
    else:
        apache_post_operative = np.nan

    if st.checkbox("Do you have patient arf_apache?", False):
        arf_apache = st.number_input('ARF APACHE:', 
                                    min_value=data.describe()["arf_apache"].loc["min"], 
                                    max_value=data.describe()["arf_apache"].loc["max"], 
                                    value=data.describe()["arf_apache"].loc["50%"])
    else:
        arf_apache = np.nan

    if st.checkbox("Do you have patient fio2_apache?", False):
        fio2_apache = st.number_input('FIO2 APACHE:', 
                                    min_value=data.describe()["fio2_apache"].loc["min"], 
                                    max_value=data.describe()["fio2_apache"].loc["max"], 
                                    value=data.describe()["fio2_apache"].loc["50%"])
    else:
        fio2_apache = np.nan

    if st.checkbox("Do you have patient map_apache?", False):
        map_apache = st.number_input('MAP APACHE:', 
                                    min_value=data.describe()["map_apache"].loc["min"], 
                                    max_value=data.describe()["map_apache"].loc["max"], 
                                    value=data.describe()["map_apache"].loc["50%"])
    else:
        map_apache = np.nan

    if st.checkbox("Do you have patient ph_apache?", False):
        ph_apache = st.number_input('PH APACHE:', 
                                    min_value=data.describe()["ph_apache"].loc["min"], 
                                    max_value=data.describe()["ph_apache"].loc["max"], 
                                    value=data.describe()["ph_apache"].loc["50%"])
    else:
        ph_apache = np.nan

    if st.checkbox("Do you have patient temp_apache?", False):
        temp_apache = st.number_input('Temp APACHE:', 
                                    min_value=data.describe()["temp_apache"].loc["min"], 
                                    max_value=data.describe()["temp_apache"].loc["max"], 
                                    value=data.describe()["temp_apache"].loc["50%"])
    else:
        temp_apache = np.nan

    if st.checkbox("Do you have patient ventilated_apache?", False):
        ventilated_apache = st.number_input('Ventilated APACHE:', 
                                            min_value=data.describe()["ventilated_apache"].loc["min"], 
                                            max_value=data.describe()["ventilated_apache"].loc["max"], 
                                            value=data.describe()["ventilated_apache"].loc["50%"])
    else:
        ventilated_apache = np.nan

    if st.checkbox("Do you have patient wbc_apache?", False):
        wbc_apache = st.number_input('WBC APACHE:', 
                                    min_value=data.describe()["wbc_apache"].loc["min"], 
                                    max_value=data.describe()["wbc_apache"].loc["max"], 
                                    value=data.describe()["wbc_apache"].loc["50%"])
    else:
        wbc_apache = np.nan

    if st.checkbox("Do you have patient d1_heartrate_max?", False):
        d1_heartrate_max = st.number_input('D1 Heartrate Max:', 
                                        min_value=data.describe()["d1_heartrate_max"].loc["min"], 
                                        max_value=data.describe()["d1_heartrate_max"].loc["max"], 
                                        value=data.describe()["d1_heartrate_max"].loc["50%"])
    else:
        d1_heartrate_max = np.nan

    if st.checkbox("Do you have patient d1_heartrate_min?", False):
        d1_heartrate_min = st.number_input('D1 Heartrate Min:', 
                                        min_value=data.describe()["d1_heartrate_min"].loc["min"], 
                                        max_value=data.describe()["d1_heartrate_min"].loc["max"], 
                                        value=data.describe()["d1_heartrate_min"].loc["50%"])
    else:
        d1_heartrate_min = np.nan

    if st.checkbox("Do you have patient d1_sysbp_invasive_min?", False):
        d1_sysbp_invasive_min = st.number_input('D1 SysBP Invasive Min:', 
                                                min_value=data.describe()["d1_sysbp_invasive_min"].loc["min"], 
                                                max_value=data.describe()["d1_sysbp_invasive_min"].loc["max"], 
                                                value=data.describe()["d1_sysbp_invasive_min"].loc["50%"])
    else:
        d1_sysbp_invasive_min = np.nan

    if st.checkbox("Do you have patient h1_heartrate_max?", False):
        h1_heartrate_max = st.number_input('H1 Heartrate Max:', 
                                        min_value=data.describe()["h1_heartrate_max"].loc["min"], 
                                        max_value=data.describe()["h1_heartrate_max"].loc["max"], 
                                        value=data.describe()["h1_heartrate_max"].loc["50%"])
    else:
        h1_heartrate_max = np.nan

    if st.checkbox("Do you have patient h1_heartrate_min?", False):
        h1_heartrate_min = st.number_input('H1 Heartrate Min:', 
                                        min_value=data.describe()["h1_heartrate_min"].loc["min"], 
                                        max_value=data.describe()["h1_heartrate_min"].loc["max"], 
                                        value=data.describe()["h1_heartrate_min"].loc["50%"])
    else:
        h1_heartrate_min = np.nan

    if st.checkbox("Do you have patient d1_arterial_ph_max?", False):
        d1_arterial_ph_max = st.number_input('D1 Arterial PH Max:', 
                                            min_value=data.describe()["d1_arterial_ph_max"].loc["min"], 
                                            max_value=data.describe()["d1_arterial_ph_max"].loc["max"], 
                                            value=data.describe()["d1_arterial_ph_max"].loc["50%"])
    else:
        d1_arterial_ph_max = np.nan

    if st.checkbox("Do you have patient d1_arterial_ph_min?", False):
        d1_arterial_ph_min = st.number_input('D1 Arterial PH Min:', 
                                            min_value=data.describe()["d1_arterial_ph_min"].loc["min"], 
                                            max_value=data.describe()["d1_arterial_ph_min"].loc["max"], 
                                            value=data.describe()["d1_arterial_ph_min"].loc["50%"])
    else:
        d1_arterial_ph_min = np.nan

    if st.checkbox("Do you have patient apache_3j_score?", False):
        apache_3j_score = st.number_input('APACHE 3J Score:', 
                                        min_value=data.describe()["apache_3j_score"].loc["min"], 
                                        max_value=data.describe()["apache_3j_score"].loc["max"], 
                                        value=data.describe()["apache_3j_score"].loc["50%"])
    else:
        apache_3j_score = np.nan

    if st.checkbox("Do you have patient apsiii?", False):
        apsiii = st.number_input('APSIII:', 
                                min_value=data.describe()["apsiii"].loc["min"], 
                                max_value=data.describe()["apsiii"].loc["max"], 
                                value=data.describe()["apsiii"].loc["50%"])
    else:
        apsiii = np.nan   

    
    


    output = ""

    input_dict = {
    'bmi': bmi,
    'hospital_los_days': hospital_los_days,
    'icu_los_days': icu_los_days,
    'albumin_apache': albumin_apache,
    'apache_2_diagnosis': apache_2_diagnosis,
    'apache_3j_diagnosis': apache_3j_diagnosis,
    'apache_post_operative': apache_post_operative,
    'arf_apache': arf_apache,
    'fio2_apache': fio2_apache,
    'map_apache': map_apache,
    'ph_apache': ph_apache,
    'temp_apache': temp_apache,
    'ventilated_apache': ventilated_apache,
    'wbc_apache': wbc_apache,
    'd1_heartrate_max': d1_heartrate_max,
    'd1_heartrate_min': d1_heartrate_min,
    'd1_sysbp_invasive_min': d1_sysbp_invasive_min,
    'h1_heartrate_max': h1_heartrate_max,
    'h1_heartrate_min': h1_heartrate_min,
    'd1_arterial_ph_max': d1_arterial_ph_max,
    'd1_arterial_ph_min': d1_arterial_ph_min,
    'apache_3j_score': apache_3j_score,
    'apsiii': apsiii,
    'gender_M': cat_map[gender_M],
    'gender_F': cat_map[gender_F],

}
    



    input_df = pd.DataFrame([input_dict])
    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
        print(output)
        if output == 1:
            st.warning("Patient have high chance of mortality" )
        else:
            st.success( "Patient is fine!!!")


if __name__ == '__main__':
    main()



# import numpy as np
# from pycaret.classification import load_model, predict_model
# import streamlit as st
# import pandas as pd

# st.set_page_config(layout="wide")
# from PIL import Image

# model = load_model('Model/lightgbm_Gosis_model')

# cat_map = {
#     "No": 0,
#     "Yes": 1,
#     "Not Available": np.nan,
#     "Male": 1,
#     "Female": 0
# }

# def predict(model, input_df):
#     predictions_df = predict_model(estimator=model, data=input_df)
#     predictions = predictions_df['prediction_label'][0]
#     return predictions

# def get_data():
#     data = pd.read_csv("gossis-1/gossis-1-eicu-only/gosis-1-24hr.csv")
#     data.columns = list(map(str.strip, list(data.columns)))
#     data = data[['bmi', 'hospital_los_days', 'icu_los_days', 'albumin_apache', 'apache_2_diagnosis', 'apache_3j_diagnosis', 'apache_post_operative', 'arf_apache', 'fio2_apache', 'map_apache', 'ph_apache', 'temp_apache', 'ventilated_apache', 'wbc_apache', 'd1_heartrate_max', 'd1_heartrate_min', 'd1_sysbp_invasive_min', 'h1_heartrate_max', 'h1_heartrate_min', 'd1_arterial_ph_max', 'd1_arterial_ph_min', 'apache_3j_score', 'apsiii']]
#     return data

# def main():
#     data = get_data()
#     st.title("ICU Prediction V2")
#     st.sidebar.title("Check Analysis:")

#     gender_M = st.selectbox('Male:', ["Not Available", "Yes", "No"])
#     gender_F = st.selectbox('Female:', ["Not Available", "Yes", "No"])

#     bmi = st.number_input('BMI:', value=np.nan if not st.checkbox("Do you have patient BMI?") else data['bmi'].median())
#     hospital_los_days = st.number_input('Hospital LOS Days:', value=np.nan if not st.checkbox("Do you have patient hospital_los_days?") else data['hospital_los_days'].median())
#     icu_los_days = st.number_input('ICU LOS Days:', value=np.nan if not st.checkbox("Do you have patient icu_los_days?") else data['icu_los_days'].median())
#     albumin_apache = st.number_input('Albumin APACHE:', value=np.nan if not st.checkbox("Do you have patient albumin_apache?") else data['albumin_apache'].median())
#     apache_2_diagnosis = st.number_input('APACHE 2 Diagnosis:', value=np.nan if not st.checkbox("Do you have patient apache_2_diagnosis?") else data['apache_2_diagnosis'].median())
#     apache_3j_diagnosis = st.number_input('APACHE 3J Diagnosis:', value=np.nan if not st.checkbox("Do you have patient apache_3j_diagnosis?") else data['apache_3j_diagnosis'].median())
#     apache_post_operative = st.number_input('APACHE Post Operative:', value=np.nan if not st.checkbox("Do you have patient apache_post_operative?") else data['apache_post_operative'].median())
#     arf_apache = st.number_input('ARF APACHE:', value=np.nan if not st.checkbox("Do you have patient arf_apache?") else data['arf_apache'].median())
#     fio2_apache = st.number_input('FIO2 APACHE:', value=np.nan if not st.checkbox("Do you have patient fio2_apache?") else data['fio2_apache'].median())
#     map_apache = st.number_input('MAP APACHE:', value=np.nan if not st.checkbox("Do you have patient map_apache?") else data['map_apache'].median())
#     ph_apache = st.number_input('PH APACHE:', value=np.nan if not st.checkbox("Do you have patient ph_apache?") else data['ph_apache'].median())
#     temp_apache = st.number_input('Temp APACHE:', value=np.nan if not st.checkbox("Do you have patient temp_apache?") else data['temp_apache'].median())
#     ventilated_apache = st.number_input('Ventilated APACHE:', value=np.nan if not st.checkbox("Do you have patient ventilated_apache?") else data['ventilated_apache'].median())
#     wbc_apache = st.number_input('WBC APACHE:', value=np.nan if not st.checkbox("Do you have patient wbc_apache?") else data['wbc_apache'].median())
#     d1_heartrate_max = st.number_input('D1 Heartrate Max:', value=np.nan if not st.checkbox("Do you have patient d1_heartrate_max?") else data['d1_heartrate_max'].median())
#     d1_heartrate_min = st.number_input('D1 Heartrate Min:', value=np.nan if not st.checkbox("Do you have patient d1_heartrate_min?") else data['d1_heartrate_min'].median())
#     d1_sysbp_invasive_min = st.number_input('D1 SysBP Invasive Min:', value=np.nan if not st.checkbox("Do you have patient d1_sysbp_invasive_min?") else data['d1_sysbp_invasive_min'].median())
#     h1_heartrate_max = st.number_input('H1 Heartrate Max:', value=np.nan if not st.checkbox("Do you have patient h1_heartrate_max?") else data['h1_heartrate_max'].median())
#     h1_heartrate_min = st.number_input('H1 Heartrate Min:', value=np.nan if not st.checkbox("Do you have patient h1_heartrate_min?") else data['h1_heartrate_min'].median())
#     d1_arterial_ph_max = st.number_input('D1 Arterial PH Max:', value=np.nan if not st.checkbox("Do you have patient d1_arterial_ph_max?") else data['d1_arterial_ph_max'].median())
#     d1_arterial_ph_min = st.number_input('D1 Arterial PH Min:', value=np.nan if not st.checkbox("Do you have patient d1_arterial_ph_min?") else data['d1_arterial_ph_min'].median())
#     apache_3j_score = st.number_input('APACHE 3J Score:', value=np.nan if not st.checkbox("Do you have patient apache_3j_score?") else data['apache_3j_score'].median())
#     apsiii = st.number_input('APSIII:', value=np.nan if not st.checkbox("Do you have patient apsiii?") else data['apsiii'].median())

#     input_dict = {
#         'bmi': bmi,
#         'hospital_los_days': hospital_los_days,
#         'icu_los_days': icu_los_days,
#         'albumin_apache': albumin_apache,
#         'apache_2_diagnosis': apache_2_diagnosis,
#         'apache_3j_diagnosis': apache_3j_diagnosis,
#         'apache_post_operative': apache_post_operative,
#         'arf_apache': arf_apache,
#         'fio2_apache': fio2_apache,
#         'map_apache': map_apache,
#         'ph_apache': ph_apache,
#         'temp_apache': temp_apache,
#         'ventilated_apache': ventilated_apache,
#         'wbc_apache': wbc_apache,
#         'd1_heartrate_max': d1_heartrate_max,
#         'd1_heartrate_min': d1_heartrate_min,
#         'd1_sysbp_invasive_min': d1_sysbp_invasive_min,
#         'h1_heartrate_max': h1_heartrate_max,
#         'h1_heartrate_min': h1_heartrate_min,
#         'd1_arterial_ph_max': d1_arterial_ph_max,
#         'd1_arterial_ph_min': d1_arterial_ph_min,
#         'apache_3j_score': apache_3j_score,
#         'apsiii': apsiii,
#         'gender_M': cat_map[gender_M],
#         'gender_F': cat_map[gender_F],
#     }

#     input_df = pd.DataFrame([input_dict])
    
#     if st.button("Predict"):
#         try:
#             output = predict(model=model, input_df=input_df)
#             if output == 1:
#                 st.warning("Patient needs to be in ICU !!!")
#             else:
#                 st.success("Patient is fine, not recommended for ICU !!!")
#         except Exception as e:
#             st.error(f"An error occurred: {e}")

# if __name__ == '__main__':
#     main()
