from datetime import date
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler

from static.classifiermodels.classifiers import get_all_trained_models


def load_application_train():
    data = pd.read_csv("C:\\Users\\deniz\\NYPD_Arrest_Data__Year_to_Date_.csv")
    return data


def convert_from_date_to_season(data_frame):
    data_frame['ARREST_DATE'] = pd.to_datetime(data_frame['ARREST_DATE'])
    data_frame['Season'] = data_frame['ARREST_DATE'].dt.month.apply(lambda x: 'Winter' if x in [12, 1, 2] else
    'Spring' if x in [3, 4, 5] else
    'Summer' if x in [6, 7, 8] else 'Fall')


def convert_from_age_group_to_categorical_age(data_frame):
    list_of_categorical_ages = []
    for i in data_frame["AGE_GROUP"]:
        match i:
            case '<18':
                list_of_categorical_ages.append('Child')
            case '18-24':
                list_of_categorical_ages.append('Young')
            case '25-44':
                list_of_categorical_ages.append('Adult')
            case '45-64':
                list_of_categorical_ages.append('Middle Aged')
            case '65+':
                list_of_categorical_ages.append('Old')
            case _:
                print("Default is working...")
    data_frame["Cat_Age"] = list_of_categorical_ages


def remove_outliers(data_frame, columns, threshold=3):
    # calculate the z-scores of  specified columns
    z_scores = np.abs(stats.zscore(data_frame[columns]))

    # if absolute z-score is greater than threshold , it is marked
    outliers = np.where(z_scores > threshold)

    # And then, all outliers are deleted
    cleaned_df = data_frame.drop(data_frame.index[outliers[0]])

    return cleaned_df


def grab_col_names(data_frame, categorical_threshold=10, cardinal_threshold=25):
    categorical_cols = [col for col in data_frame.columns if data_frame[col].dtype == "O"]

    # Numerical columns that behave like categorical based on the threshold categorical_threshold
    numeric_looking_but_categorical = [col for col in data_frame.columns if
                                       data_frame[col].dtype != "O" and data_frame[
                                           col].nunique() < categorical_threshold]

    # Categorical columns that behave like numerical based on the threshold cardinal_threshold
    categorical_looking_but_cardinal = [col for col in data_frame.columns if
                                        data_frame[col].dtype == "O" and data_frame[col].nunique() > cardinal_threshold]

    categorical_cols = categorical_cols + numeric_looking_but_categorical

    categorical_cols = [col for col in categorical_cols if col not in categorical_looking_but_cardinal]

    numeric_cols = [col for col in data_frame.columns if data_frame[col].dtype != "O"]

    numeric_cols = [col for col in numeric_cols if col not in numeric_looking_but_categorical]

    print(f"Observations: {data_frame.shape[0]}")
    print(f"Variables: {data_frame.shape[1]}")
    print(f"Categorical Columns: {len(categorical_cols)}")
    print(f"Numeric Columns: {len(numeric_cols)}")
    print(f"Categorical Looking but Cardinal: {len(categorical_looking_but_cardinal)}")
    print(f"Numeric Looking but Categorical: {len(numeric_looking_but_categorical)}")
    return categorical_cols, numeric_cols, categorical_looking_but_cardinal


def label_encoder(data_frame, binary_col):
    le = LabelEncoder()
    data_frame[binary_col] = le.fit_transform(data_frame[binary_col])
    return data_frame


def one_hot_encoder(data_frame, categorical_cols, drop_first=False):
    data_frame = pd.get_dummies(data_frame, columns=categorical_cols, drop_first=drop_first)
    return data_frame


def prepare_new_data(new_data):
    converted_season = date(2023, 1, 1)
    if new_data["season"] == "spring":
        converted_season = date(2023, 3, 1)
    elif new_data["season"] == "summer":
        converted_season = date(2023, 6, 1)
    elif new_data["season"] == "fall":
        converted_season = date(2023, 9, 1)
    else:
        converted_season = date(2023, 12, 1)

    dict_row = {
        "ARREST_KEY" : 273821910,
        "ARREST_DATE" : converted_season,
        "PD_CD" : int(new_data["pdcd"]),
        "PD_DESC" : "ASSAULT 3",
        "KY_CD" : float(new_data["kycd"]),
        "OFNS_DESC" : "ASSAULT 3 & RELATED OFFENSES",
        "LAW_CODE" : "PL 1200001",
        "LAW_CAT_CD" : (str(new_data["offense"]))[0].capitalize(),
        "ARREST_BORO" : (str(new_data["borough"])).upper(),
        "ARREST_PRECINCT" : int(new_data["precinct"]),
        "JURISDICTION_CODE" : int(new_data["jurisdiction"]),
        "AGE_GROUP" : new_data["age"],
        "PERP_SEX" : (str(new_data["gender"]))[0].capitalize(),
        "PERP_RACE" : (str(new_data["race"])).upper(),
        "X_COORD_CD" : 1002760,
        "Y_COORD_CD" : 193531,
        "Latitude" : float(new_data["latitude"]),
        "Longitude" : float(new_data["longitude"]),
        "New Georeferenced Column" : "POINT (-73.9332467142579 40.69785526209324)",
        "Community Districts" : 42.000,
        "Borough Boundaries" : 2.000,
        "City Council Districts" : 30.000,
        "Police Precincts" : 53.000,
        "Zip Codes" : 18181.000,
        }
    print("Done Dist of Row: ", dict_row)
    df2 = pd.DataFrame(dict_row, index=[0])
    return df2


def make_preprocessing(request):
    df1 = load_application_train()
    df2 = prepare_new_data(request.POST)
    df = pd.concat([df1, df2], ignore_index=True)
    convert_from_date_to_season(df)
    convert_from_age_group_to_categorical_age(df)
    df.drop("ARREST_KEY", axis=1, inplace=True)
    df.drop("ARREST_DATE", axis=1, inplace=True)
    df.drop("AGE_GROUP", axis=1, inplace=True)
    df.drop("New Georeferenced Column", axis=1, inplace=True)
    df.drop("Community Districts", axis=1, inplace=True)
    df.drop("Borough Boundaries", axis=1, inplace=True)
    df.drop("City Council Districts", axis=1, inplace=True)
    df.drop("Police Precincts", axis=1, inplace=True)
    df.drop("Zip Codes", axis=1, inplace=True)
    df.drop("PD_DESC", axis=1, inplace=True)
    df.drop("OFNS_DESC", axis=1, inplace=True)
    df.drop("X_COORD_CD", axis=1, inplace=True)
    df.drop("Y_COORD_CD", axis=1, inplace=True)
    df = df[df['LAW_CAT_CD'].isin(['F', 'M', 'V'])]
    df = df[df['PERP_SEX'].isin(['F', 'M'])]
    df = df.dropna()  # Delete all None values
    df = remove_outliers(df, columns=['Latitude', 'Longitude'])
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    df[num_cols].corr()
    df = df[df['LAW_CODE'].str.startswith('PL')]
    df['LAW_CODE'] = df['LAW_CODE'].str[:6]
    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                   and df[col].nunique() == 2]
    print(binary_cols)
    for col in binary_cols:
        label_encoder(df, col)

    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    df = one_hot_encoder(df, cat_cols, drop_first=False)
    scale = StandardScaler()
    df[num_cols] = scale.fit_transform(df[num_cols])
    #df = df.sample(frac=1, random_state=42)

    #random_indexes = df.sample(n=4500, replace=False, random_state=42).index
    #df = df.loc[random_indexes]
    print(len(df))
    print(df.tail(10))
    return df


def get_test_result(request):
    print("Dict of Request: ", request.POST)
    df = make_preprocessing(request)
    processed_version_of_the_added_data = df.iloc[- 1]
    law = make_test_given_row(processed_version_of_the_added_data)
    return law


def make_test_given_row(test_data):
    all_classifiers = get_all_trained_models()
    decision_tree = all_classifiers["decision_tree"]

    columns = [
        'PD_CD', 'KY_CD', 'ARREST_PRECINCT', 'JURISDICTION_CODE',
        'Latitude', 'Longitude', 'LAW_CAT_CD_F', 'LAW_CAT_CD_M',
        'LAW_CAT_CD_V', 'ARREST_BORO_B', 'ARREST_BORO_K',
        'ARREST_BORO_M', 'ARREST_BORO_Q', 'ARREST_BORO_S',
        'PERP_RACE_AMERICAN INDIAN/ALASKAN NATIVE', 'PERP_RACE_ASIAN / PACIFIC ISLANDER', 'PERP_RACE_BLACK',
        'PERP_RACE_BLACK HISPANIC', 'PERP_RACE_UNKNOWN', 'PERP_RACE_WHITE',
        'PERP_RACE_WHITE HISPANIC', 'Season_Fall', 'Season_Spring',
        'Season_Summer', 'Season_Winter', 'Cat_Age_Adult',
        'Cat_Age_Child', 'Cat_Age_Middle Aged', 'Cat_Age_Old',
        'Cat_Age_Young', 'PERP_SEX_0', 'PERP_SEX_1'
    ]

    test_rows = []
    test_row = []
    if len(test_data) != 30:
        for key in test_data.keys():
            if key in columns:
                test_row.append(test_data[key])
        test_rows.append(test_row)

    
    print(test_rows[0])
    prediction = decision_tree.predict(test_rows)
    print("Prediction:", prediction)
    return prediction[0]