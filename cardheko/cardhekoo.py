import pandas as pd
import re
import ast
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


file_path = 'C:/Users/bhavan/Downloads/kolkataa_cars.xlsx'
kolkataa_cars_df = pd.read_excel(file_path)

# Extract the dictionary values from the new_car_detail column
kolkataa_cars_df['new_car_detail'] = kolkataa_cars_df['new_car_detail'].apply(eval)

# Normalize the dictionary values into separate columns
details_df = pd.json_normalize(kolkataa_cars_df['new_car_detail'])

# Concatenate the new columns with the original DataFrame
kolkataa_cars_df = pd.concat([kolkataa_cars_df.drop(columns=['new_car_detail']), details_df], axis=1)

# Function to clean price strings
def clean_price(price):
    if isinstance(price, str):
        # Remove currency symbols and text
        price = re.sub(r'[^\d.]', '', price)
        # Convert to float
        return float(price) if price else None
    return price

# Convert km to float64
kolkataa_cars_df['km'] = kolkataa_cars_df['km'].str.replace(',', '').astype(float)

# Clean and convert price columns to float64
kolkataa_cars_df['price'] = kolkataa_cars_df['price'].apply(clean_price)
kolkataa_cars_df['priceActual'] = kolkataa_cars_df['priceActual'].apply(clean_price)
kolkataa_cars_df['priceSaving'] = kolkataa_cars_df['priceSaving'].apply(clean_price)

kolkataa_cars_df.drop(columns=['km'], inplace=True)

# Drop the specified columns
kolkataa_cars_df = kolkataa_cars_df.drop(columns=['priceActual', 'priceSaving', 'priceFixedText'])

#one heart encoding 
if 'ft' in kolkataa_cars_df.columns:
    kolkataa_cars_df = pd.get_dummies(kolkataa_cars_df, columns=['ft'], prefix='ft')
    
# Convert all boolean columns in kolkata_cars_df to integers
kolkataa_cars_df = kolkataa_cars_df.astype({col: 'int' for col in kolkataa_cars_df.select_dtypes('bool').columns})

kolkataa_cars_df = kolkataa_cars_df.drop(columns=['owner'])


def str_to_dict(row):
    return ast.literal_eval(row)

# Apply the function to the Series
data_dicts = kolkataa_cars_df["new_car_overview"].apply(str_to_dict)

# Now, extract the 'top' data from each dictionary and create a DataFrame
data_list = [{item['key']: item['value'] for item in d['top']} for d in data_dicts]

# Convert the list of dictionaries into a DataFrame
df_overview = pd.DataFrame(data_list)

# Concatenate the new columns with the original DataFrame
kolkataa_cars_df = pd.concat([kolkataa_cars_df, df_overview], axis=1)

kolkataa_cars_df['Kms Driven'] = kolkataa_cars_df['Kms Driven'].replace(r'\s*Kms', '', regex=True)

# Function to extract the year from the 'Registration Year' column
def extract_year(value):
    if pd.isna(value):
        return np.nan
    try:
        return int(value)
    except ValueError:
        # Extract the year from strings like 'Jul 2017'
        return int(value.split()[-1])

# Change the data type of 'Registration Year' to integer
if 'Registration Year' in kolkataa_cars_df.columns:
    kolkataa_cars_df['Registration Year'] = kolkataa_cars_df['Registration Year'].apply(extract_year).astype('Int64')

kolkataa_cars_df['RTO'] = kolkataa_cars_df['RTO'].fillna('not available')

kolkataa_cars_df = kolkataa_cars_df.drop(columns=['Ownership'])

# Remove the text using regular expressions
kolkataa_cars_df['Seats'] = kolkataa_cars_df['Seats'].str.replace(r'\b\d+\s*seats\b', '', regex=True, case=False).str.strip()

kolkataa_cars_df = kolkataa_cars_df.loc[:, kolkataa_cars_df.columns != 'Seats']

kolkataa_cars_df = kolkataa_cars_df.loc[:, ~kolkataa_cars_df.columns.duplicated(keep='first')]

kolkataa_cars_df['Engine Displacement'] = kolkataa_cars_df['Engine Displacement'].str.extract('(\d+)')

# changing regitration year to int doubt in this and seat also and RTO

# Convert the string to a dictionary
def str_to_dict(row):
    return ast.literal_eval(row)

# Apply the function to the 'new_car_specs' column
data_dicts_specs = kolkataa_cars_df["new_car_specs"].apply(str_to_dict)

# Extract the relevant data from each dictionary
data_list_specs = []
for d in data_dicts_specs:
    # Extract 'top' key-value pairs
    top_dict = {item['key']: item['value'] for item in d['top']}
    
    # Extract 'data' sections key-value pairs
    for section in d['data']:
        section_dict = {item['key']: item['value'] for item in section['list']}
        top_dict.update(section_dict)
    
    data_list_specs.append(top_dict)

# Create a DataFrame from the extracted data
df_specs = pd.DataFrame(data_list_specs)

# Concatenate the new columns with the original DataFrame
kolkataa_cars_df = pd.concat([kolkataa_cars_df, df_specs], axis=1)

# Convert the string to a dictionary
def str_to_dict(row):
    return ast.literal_eval(row)

# Apply the function to the 'new_car_specs' column
data_dicts_specs = kolkataa_cars_df["new_car_specs"].apply(str_to_dict)

# Extract the relevant data from each dictionary
data_list_specs = [{item['key']: item['value'] for item in d['top']} for d in data_dicts_specs]
df_specs = pd.DataFrame(data_list_specs)

# Concatenate the new columns with the original DataFrame
kolkataa_cars_df = pd.concat([kolkataa_cars_df, df_specs], axis=1)

# Remove the 'Ground Clearance Unladen' column if it exists
if 'Ground Clearance Unladen' in kolkataa_cars_df.columns:
    kolkataa_cars_df = kolkataa_cars_df.drop(columns=['Ground Clearance Unladen'])

# Drop the 'Compression_Ratio' column if it exists
if 'Compression Ratio' in kolkataa_cars_df.columns:
    kolkataa_cars_df = kolkataa_cars_df.drop(columns=['Compression Ratio'])
    
if 'Compression_Ratio' in kolkataa_cars_df.columns:
    kolkataa_cars_df = kolkataa_cars_df.drop(columns=['Compression_Ratio'])

if 'Max Power' in kolkataa_cars_df.columns:
    kolkataa_cars_df = kolkataa_cars_df.drop(columns=['Max Power'])
    
kolkataa_cars_df = kolkataa_cars_df.drop(columns=['Wheel Size'])

kolkataa_cars_df = kolkataa_cars_df.drop(columns=['BoreX Stroke'])

kolkataa_cars_df = kolkataa_cars_df.drop(columns=['Front Tread'])

kolkataa_cars_df = kolkataa_cars_df.drop(columns=['Rear Tread'])

kolkataa_cars_df = kolkataa_cars_df.drop(columns=['Gross Weight'])

kolkataa_cars_df = kolkataa_cars_df.drop(columns=['Acceleration'])

kolkataa_cars_df = kolkataa_cars_df.drop(columns=['Alloy Wheel Size'])

kolkataa_cars_df = kolkataa_cars_df.drop(columns=['Max Torque'])

kolkataa_cars_df.drop(columns=['Tyre Type'], inplace=True)

kolkataa_cars_df.drop(columns=['No Door Numbers'], inplace=True)

kolkataa_cars_df.drop(columns=['Turning Radius'], inplace=True)

kolkataa_cars_df.drop(columns=['Length', 'Width'], inplace=True)

kolkataa_cars_df.drop(columns=['Values per Cylinder'], inplace=True)

kolkataa_cars_df = kolkataa_cars_df.drop(columns=['Displacement'])

kolkataa_cars_df = kolkataa_cars_df.drop(columns=['trendingText.imgUrl', 'trendingText.heading', 'trendingText.desc',])


def process_torque(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find decimal numbers or integers, allowing commas
    match = re.search(r'([\d,]+\.?\d*)', text)  # Capture numeric value with commas

    if match:
        # Return the numeric torque value
        return match.group(1).replace(',', '')  # Optionally remove commas for numerical processing
   
# Process each row in the 'Torque' column
for i in range(len(kolkataa_cars_df)):
    kolkataa_cars_df.at[i, 'Torque'] = process_torque(kolkataa_cars_df.at[i, 'Torque'])


def process_milage(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find decimal numbers or integers, allowing commas
    match = re.search(r'([\d,]+\.?\d*)', text)  # Capture numeric value with commas

    if match:
        # Return the numeric milage value
        return match.group(1).replace(',', '')  # Optionally remove commas for numerical processing
   
# Process each row in the 'Milage' column
for i in range(len(kolkataa_cars_df)):
    kolkataa_cars_df.at[i, 'Mileage'] = process_milage(kolkataa_cars_df.at[i, 'Mileage'])


def process_engine(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find decimal numbers or integers, allowing commas
    match = re.search(r'([\d,]+\.?\d*)', text)  # Capture numeric value with commas

    if match:
        # Return the numeric engine value
        return match.group(1).replace(',', '')  # Optionally remove commas for numerical processing
   

# Process each row in the 'Engine' column
for i in range(len(kolkataa_cars_df)):
    kolkataa_cars_df.at[i, 'Engine'] = process_engine(kolkataa_cars_df.at[i, 'Engine'])
    


def extract_all_numbers(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find all numeric values in the text
    matches = re.findall(r'[\d,]+\.?\d*', text)  # Capture all numeric values, allowing for commas

    # Join the found matches into a single string
    if matches:
        # Remove commas and join numbers without brackets
        return ', '.join(match.replace(',', '') for match in matches)  # Join with a comma
  

# Process each row in the 'Height' column
for i in range(len(kolkataa_cars_df)):
    kolkataa_cars_df.at[i, 'Height'] = extract_all_numbers(kolkataa_cars_df.at[i, 'Height'])
    
def extract_all_numbers(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find all numeric values in the text
    matches = re.findall(r'[\d,]+\.?\d*', text)  # Capture all numeric values, allowing for commas

    # Join the found matches into a single string
    if matches:
        # Remove commas and join numbers without brackets
        return ', '.join(match.replace(',', '') for match in matches)  # Join with a comma
    

# Process each row in the 'Wheel Base' column
for i in range(len(kolkataa_cars_df)):
    kolkataa_cars_df.at[i, 'Wheel Base'] = extract_all_numbers(kolkataa_cars_df.at[i, 'Wheel Base'])
    
def extract_all_numbers(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find all numeric values in the text
    matches = re.findall(r'[\d,]+\.?\d*', text)  # Capture all numeric values, allowing for commas

    # Join the found matches into a single string
    if matches:
        # Remove commas and join numbers without brackets
        return ', '.join(match.replace(',', '') for match in matches)  # Join with a comma
    
# Process each row in the 'Kerb Weight' column
for i in range(len(kolkataa_cars_df)):
    kolkataa_cars_df.at[i, 'Kerb Weight'] = extract_all_numbers(kolkataa_cars_df.at[i, 'Kerb Weight'])

def extract_all_numbers(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find all numeric values in the text
    matches = re.findall(r'[\d,]+\.?\d*', text)  # Capture all numeric values, allowing for commas

    # Join the found matches into a single string
    if matches:
        # Remove commas and join numbers without brackets
        return ', '.join(match.replace(',', '') for match in matches)  # Join with a comma
   

# Process each row in the 'Gear Box' column
for i in range(len(kolkataa_cars_df)):
    kolkataa_cars_df.at[i, 'Gear Box'] = extract_all_numbers(kolkataa_cars_df.at[i, 'Gear Box'])
    
def extract_all_numbers(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find all numeric values in the text
    matches = re.findall(r'[\d,]+\.?\d*', text)  # Capture all numeric values, allowing for commas

    # Join the found matches into a single string
    if matches:
        # Remove commas and join numbers without brackets
        return ', '.join(match.replace(',', '') for match in matches)  # Join with a comma

# Process each row in the 'Cargo Volume' column
for i in range(len(kolkataa_cars_df)):
    kolkataa_cars_df.at[i, 'Cargo Volumn'] = extract_all_numbers(kolkataa_cars_df.at[i, 'Cargo Volumn'])


# Find columns that appear more than once
duplicate_columns = kolkataa_cars_df.columns[kolkataa_cars_df.columns.duplicated(keep=False)]

# Drop columns that appear more than once
kolkataa_cars_df = kolkataa_cars_df.drop(columns=duplicate_columns)


def clean_weight(value):
    if pd.isnull(value):
        return None
    value = re.sub(r'[^0-9,]', '', str(value))  # Keep numbers and commas
    if ',' in value:
        value = value.split(',')[-1]  # Keep the value after the comma
    return int(value) if value.isdigit() else None

kolkataa_cars_df['Kerb Weight'] = kolkataa_cars_df['Kerb Weight'].apply(clean_weight)
kolkataa_cars_df['Height'] = kolkataa_cars_df['Height'].apply(clean_weight) 
kolkataa_cars_df['Cargo Volumn'] = kolkataa_cars_df['Cargo Volumn'].apply(clean_weight)
kolkataa_cars_df['Gear Box'] = kolkataa_cars_df['Gear Box'].apply(clean_weight)

kolkataa_cars_df['Gear Box'] = kolkataa_cars_df['Gear Box'].astype(float)
# Define the grouping column and target column
group_column = 'bt'  # Replace with your actual group column
target_column = 'Super Charger'  # Replace with the column you want to impute

# Perform group mode imputation
kolkataa_cars_df[target_column] = kolkataa_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))

# Define the grouping column and target column
group_column = 'bt'  # Replace with your actual group column
target_column = 'Turbo Charger'  # Replace with the column you want to impute

# Perform group mode imputation
kolkataa_cars_df[target_column] = kolkataa_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))

# Define the grouping column and target column
group_column = 'bt'  # Replace with your actual group column
target_column = 'Steering Type'  # Replace with the column you want to impute

# Perform group mode imputation
kolkataa_cars_df[target_column] = kolkataa_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))

# Define the grouping column and target column
group_column = 'bt'  # Replace with your actual group column
target_column = 'Engine Type'  # Replace with the column you want to impute

# Perform group mode imputation
kolkataa_cars_df[target_column] = kolkataa_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))

# Define the grouping column and target column
group_column = 'bt'  # Replace with your actual group column
target_column = 'Drive Type'  # Replace with the column you want to impute

# Perform group mode imputation
kolkataa_cars_df[target_column] = kolkataa_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))

# Define the grouping column and target column
group_column = 'bt'  # Replace with your actual group column
target_column = 'Fuel Suppy System'  # Replace with the column you want to impute

# Perform group mode imputation
kolkataa_cars_df[target_column] = kolkataa_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))

# Define the grouping column and target column
group_column = 'bt'  # Replace with your actual group column
target_column = 'Cargo Volumn'  # Replace with the column you want to impute

# Perform group mean imputation
kolkataa_cars_df[target_column] = kolkataa_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(pd.to_numeric(x, errors='coerce').mean() if not x.empty else 0))

# Define the grouping column and target column
group_column = 'bt'  # Replace with your actual group column
target_column = 'Cargo Volumn'  # Replace with the column you want to impute

# Perform group mean imputation
kolkataa_cars_df[target_column] = kolkataa_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(pd.to_numeric(x, errors='coerce').mean() if not x.empty else 0))


# Define the grouping column and target column
group_column = 'bt'  # Replace with your actual group column
target_column = 'Gear Box'  # Replace with the column you want to impute

# Perform group mean imputation
kolkataa_cars_df[target_column] = kolkataa_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(np.ceil(pd.to_numeric(x, errors='coerce').mean())) if not x.empty else 0)


# Define the grouping column and target column
group_column = 'bt'  # Replace with your actual group column
target_column = 'Height'  # Replace with the column you want to impute

# Perform group mean imputation
kolkataa_cars_df[target_column] = kolkataa_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(pd.to_numeric(x, errors='coerce').mean() if not x.empty else 0))

# Define the grouping column and target columns
group_column = 'bt'  # The column used for grouping
target_columns = ['No of Cylinder']  # List of columns to impute

kolkataa_cars_df['No of Cylinder'] = kolkataa_cars_df.groupby('bt')['No of Cylinder'].transform(lambda x: x.fillna(np.ceil(pd.to_numeric(x, errors='coerce').mean()) if not x.empty else 0))

# Define the grouping column and target column
group_column = 'bt'  # Replace with your actual group column
target_column = 'Value Configuration'  # Replace with the column you want to impute

# Perform group mode imputation
kolkataa_cars_df[target_column] = kolkataa_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))

# Define the grouping column and target column
group_column = 'bt'  # Replace with your actual group column
target_column = 'Kerb Weight'  # Replace with the column you want to impute

# Perform group mean imputation
kolkataa_cars_df[target_column] = kolkataa_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(pd.to_numeric(x, errors='coerce').mean() if not x.empty else 0))

# Define the grouping column and target column
group_column = 'bt'  # Replace with your actual group column
target_column = 'Seating Capacity'  # Replace with the column you want to impute

# Perform group mean imputation
kolkataa_cars_df[target_column] = kolkataa_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(np.ceil(pd.to_numeric(x, errors='coerce').mean())) if not x.empty else 0)


if 'Top Speed' in kolkataa_cars_df.columns:
    kolkataa_cars_df.drop(columns=['Top Speed'], inplace=True)
 
kolkataa_cars_df.drop(columns=['Wheel Base'], inplace=True)

kolkataa_cars_df.drop(columns=['Front Brake Type', 'Rear Brake Type'], inplace=True)

group_column = 'bt'  # Column to group by
target_columns = ['Registration Year', 'Year of Manufacture']  # Columns to impute

kolkataa_cars_df[['Registration Year', 'Year of Manufacture']] = kolkataa_cars_df.groupby('bt')[['Registration Year', 'Year of Manufacture']].transform(lambda x: x.fillna(np.ceil(pd.to_numeric(x, errors='coerce').mean()) if not x.empty else 0))

kolkataa_cars_df = kolkataa_cars_df.dropna(subset=['Cargo Volumn'])

kolkataa_cars_df = kolkataa_cars_df.drop(columns=['new_car_overview', 'new_car_feature', 'new_car_specs', 'car_links',"Fuel Type","Transmission"])



kolkataa_cars_df['Kms Driven'] = kolkataa_cars_df['Kms Driven'].str.replace(' km', '').str.replace(',', '').astype(float)
kolkataa_cars_df['Engine Displacement'] = kolkataa_cars_df['Engine Displacement'].astype(float)
kolkataa_cars_df[['Kerb Weight', 'Height']] = kolkataa_cars_df[['Kerb Weight', 'Height']].astype(float)
kolkataa_cars_df[['Seating Capacity', 'Cargo Volumn']] = kolkataa_cars_df[['Seating Capacity', 'Cargo Volumn']].astype(float)

#ont hot encoding
kolkataa_cars_df = pd.get_dummies(kolkataa_cars_df, columns=['bt'], prefix='bt')
kolkataa_cars_df[['bt_Convertibles', 'bt_Coupe', 'bt_Hatchback', 'bt_MUV', 'bt_Minivans', 'bt_SUV', 'bt_Sedan']] = kolkataa_cars_df[['bt_Convertibles', 'bt_Coupe', 'bt_Hatchback', 'bt_MUV', 'bt_Minivans', 'bt_SUV', 'bt_Sedan']].apply(lambda x: x.astype(int))

kolkataa_cars_df = pd.get_dummies(kolkataa_cars_df, columns=['transmission'], prefix='transmission')
kolkataa_cars_df[['transmission_Automatic', 'transmission_Manual']] = kolkataa_cars_df[['transmission_Automatic', 'transmission_Manual']].astype(int)

# Initialize the LabelEncoder
le = LabelEncoder()

# Apply Label Encoding to the 'oem' column
kolkataa_cars_df['oem'] = le.fit_transform(kolkataa_cars_df['oem'])
kolkataa_cars_df['model'] = le.fit_transform(kolkataa_cars_df['model'])
kolkataa_cars_df['variantName'] = le.fit_transform(kolkataa_cars_df['variantName'])
kolkataa_cars_df['RTO'] = le.fit_transform(kolkataa_cars_df['RTO'])
kolkataa_cars_df['Color'] = le.fit_transform(kolkataa_cars_df['Color'])
kolkataa_cars_df['Engine Type'] = le.fit_transform(kolkataa_cars_df['Engine Type'])
kolkataa_cars_df['Value Configuration'] = le.fit_transform(kolkataa_cars_df['Value Configuration'])
kolkataa_cars_df['Fuel Suppy System'] = le.fit_transform(kolkataa_cars_df['Fuel Suppy System'])
kolkataa_cars_df['Insurance Validity'] = le.fit_transform(kolkataa_cars_df['Insurance Validity'])
kolkataa_cars_df['Turbo Charger'] = le.fit_transform(kolkataa_cars_df['Turbo Charger'])
kolkataa_cars_df['Super Charger'] = le.fit_transform(kolkataa_cars_df['Super Charger'])
kolkataa_cars_df['Drive Type'] = le.fit_transform(kolkataa_cars_df['Drive Type'])
kolkataa_cars_df['Steering Type'] = le.fit_transform(kolkataa_cars_df['Steering Type'])

# Apply Min-Max scaling to the 'ownerNo' column
scaler = MinMaxScaler()
kolkataa_cars_df['ownerNo'] = scaler.fit_transform(kolkataa_cars_df[['ownerNo']])
# List of all numerical columns for Min-Max scaling
numerical_columns = [
    'price', 'ft_Cng', 'ft_Diesel', 'ft_Electric', 'ft_Lpg', 'ft_Petrol', 
    'Registration Year', 'Kms Driven', 'Engine Displacement', 'Year of Manufacture', 
    'Height', 'Kerb Weight', 'Seating Capacity', 'Cargo Volumn', 
    'bt_Convertibles', 'bt_Coupe', 'bt_Hatchback', 'bt_MUV', 'bt_Minivans', 
    'bt_SUV', 'bt_Sedan', 'transmission_Automatic', 'transmission_Manual',"oem","model","modelYear","centralVariantId","variantName",
    "price","Insurance Validity","RTO","Color","Engine Type","No of Cylinder","Value Configuration","Fuel Suppy System","Turbo Charger","Super Charger",
    "Gear Box","Drive Type","Steering Type"

]

# Apply Min-Max scaling to all numerical columns
kolkataa_cars_df[numerical_columns] = scaler.fit_transform(kolkataa_cars_df[numerical_columns])

print(kolkataa_cars_df.dtypes)

output_file_path = 'C:/Users/bhavan/Downloads/kolkataa_cars_df.csv'
kolkataa_cars_df.to_csv(output_file_path, index=False)


#going to the next city. have to perform mean and one heart encoding in the kolkataa_cars_df

file_path = 'C:/Users/bhavan/Downloads/jaipur_cars.xlsx'
jaipur_cars_df = pd.read_excel(file_path)

# Extract the dictionary values from the new_car_detail column
jaipur_cars_df['new_car_detail'] = jaipur_cars_df['new_car_detail'].apply(eval)

# Normalize the dictionary values into separate columns
details_df = pd.json_normalize(jaipur_cars_df['new_car_detail'])

# Concatenate the new columns with the original DataFrame
jaipur_cars_df = pd.concat([jaipur_cars_df.drop(columns=['new_car_detail']), details_df], axis=1)

jaipur_cars_df = jaipur_cars_df.drop(columns=['priceActual', 'priceSaving', 'priceFixedText'])

jaipur_cars_df = jaipur_cars_df.drop(columns=['owner'])

def extract_all_numbers(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find all numeric values in the text
    matches = re.findall(r'[\d,]+\.?\d*', text)  # Capture all numeric values, allowing for commas

    # Join the found matches into a single string
    if matches:
        # Remove commas and join numbers without brackets
        return ', '.join(match.replace(',', '') for match in matches)  # Join with a comma
 

# Process each row in the 'price' column
for i in range(len(jaipur_cars_df)):
    jaipur_cars_df.at[i, 'price'] = extract_all_numbers(jaipur_cars_df.at[i, 'price'])



# Function to convert string to dictionary
def str_to_dict(row):
    return ast.literal_eval(row)

# Apply the function to the Series
data_dicts = jaipur_cars_df["new_car_overview"].apply(str_to_dict)

# Extract the 'top' data from each dictionary and create a DataFrame
data_list = [{item['key']: item['value'] for item in d['top']} for d in data_dicts]

# Convert the list of dictionaries into a DataFrame
df_overview = pd.DataFrame(data_list)

# Concatenate the new columns with the original DataFrame
jaipur_cars_df = pd.concat([jaipur_cars_df, df_overview], axis=1)

# Function to extract the year from the 'Registration Year' column
def extract_year(value):
    if pd.isna(value):
        return np.nan
    try:
        return int(value)
    except ValueError:
        # Extract the year from strings like 'Jul 2017'
        return int(value.split()[-1])

# Apply the function to the 'Registration Year' column
jaipur_cars_df['Registration Year'] = jaipur_cars_df['Registration Year'].apply(extract_year)



# Define the function to convert string to dictionary
def str_to_dict(row):
    return ast.literal_eval(row)

# Apply the function to the 'new_car_specs' column in jaipur_cars_df
data_dicts_specs = jaipur_cars_df["new_car_specs"].apply(str_to_dict)

# Extract the relevant data from each dictionary
data_list_specs = []
for d in data_dicts_specs:
    # Extract 'top' key-value pairs
    top_dict = {item['key']: item['value'] for item in d['top']}
    
    # Extract 'data' sections key-value pairs
    for section in d['data']:
        section_dict = {item['key']: item['value'] for item in section['list']}
        top_dict.update(section_dict)
    
    data_list_specs.append(top_dict)

# Create a DataFrame from the extracted data
df_specs = pd.DataFrame(data_list_specs)

# Concatenate the new columns with the original DataFrame
jaipur_cars_df = pd.concat([jaipur_cars_df, df_specs], axis=1)

# Remove the 'Ground Clearance Unladen' column if it exists in jaipur_cars_df
if 'Ground Clearance Unladen' in jaipur_cars_df.columns:
    jaipur_cars_df = jaipur_cars_df.drop(columns=['Ground Clearance Unladen'])

# Drop the 'Compression Ratio' column if it exists in jaipur_cars_df
if 'Compression Ratio' in jaipur_cars_df.columns:
    jaipur_cars_df = jaipur_cars_df.drop(columns=['Compression Ratio'])
    
# Drop the 'Compression Ratio' column if it exists in jaipur_cars_df
if 'BoreX Stroke' in jaipur_cars_df.columns:
    jaipur_cars_df = jaipur_cars_df.drop(columns=['BoreX Stroke'])

# Drop the 'Compression Ratio' column if it exists in jaipur_cars_df
if 'Alloy Wheel Size' in jaipur_cars_df.columns:
    jaipur_cars_df = jaipur_cars_df.drop(columns=['Alloy Wheel Size'])
    
# Drop the 'Compression Ratio' column if it exists in jaipur_cars_df
if 'Turning Radius' in jaipur_cars_df.columns:
    jaipur_cars_df = jaipur_cars_df.drop(columns=['Turning Radius'])
    

# Drop the 'Compression Ratio' column if it exists in jaipur_cars_df
if 'Tyre Type' in jaipur_cars_df.columns:
    jaipur_cars_df = jaipur_cars_df.drop(columns=['Tyre Type'])
    
# Drop the 'Compression Ratio' column if it exists in jaipur_cars_df
if 'Gross Weight' in jaipur_cars_df.columns:
    jaipur_cars_df = jaipur_cars_df.drop(columns=['Gross Weight'])
    
# Drop the 'Compression Ratio' column if it exists in jaipur_cars_df
if 'No Door Numbers' in jaipur_cars_df.columns:
    jaipur_cars_df = jaipur_cars_df.drop(columns=['No Door Numbers'])
    
# Drop the 'Top Speed' column
jaipur_cars_df.drop(columns=['Top Speed'], inplace=True)

# Drop the 'Seats' column from jaipur_cars_df
jaipur_cars_df = jaipur_cars_df.drop(columns=['Seats'])

# Drop the 'Length' and 'Width' columns from jaipur_cars_df
jaipur_cars_df = jaipur_cars_df.drop(columns=['Length', 'Width'])

# Drop the 'Acceleration' column from jaipur_cars_df
jaipur_cars_df = jaipur_cars_df.drop(columns=['Acceleration'])

# Drop 'Front Tread' and 'Rear Tread' columns if they exist
jaipur_cars_df = jaipur_cars_df.drop(columns=['Front Tread', 'Rear Tread'], errors='ignore')

# Drop the 'Wheel Size' column from jaipur_cars_df
jaipur_cars_df = jaipur_cars_df.drop(columns=['Wheel Size'])

# Drop the 'Kms Driven' and 'Max Torque' columns from jaipur_cars_df
jaipur_cars_df = jaipur_cars_df.drop(columns=['Kms Driven', 'Max Torque'])

jaipur_cars_df.drop(columns=['Fuel Type','Engine Displacement','Max Power','Displacement','Ownership'], inplace=True)

jaipur_cars_df.drop(columns=['new_car_overview', 'new_car_feature', 'new_car_specs', 'car_links'], inplace=True)

def process_engine(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find decimal numbers or integers, allowing commas
    match = re.search(r'([\d,]+\.?\d*)', text)  # Capture numeric value with commas

    if match:
        # Return the numeric engine value, removing commas for numerical processing
        return match.group(1).replace(',', '')
   

# Process each row in the 'Engine' column of jaipur_cars_df using iloc
for i in range(len(jaipur_cars_df)):
    # Use iloc to avoid KeyError for indices that may not be continuous
    jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Engine')] = process_engine(jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Engine')])
    
# Function to process numeric values from string (for both 'Engine' and 'Mileage')
def process_numeric(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find decimal numbers or integers, allowing commas
    match = re.search(r'([\d,]+\.?\d*)', text)  # Capture numeric value with commas

    if match:
        # Return the numeric value, removing commas for numerical processing
        return match.group(1).replace(',', '')

for i in range(len(jaipur_cars_df)):
 jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Mileage')] = process_numeric(jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Mileage')])
    
def process_numeric(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find decimal numbers or integers, allowing commas
    match = re.search(r'([\d,]+\.?\d*)', text)  # Capture numeric value with commas

    if match:
        # Return the numeric value, removing commas for numerical processing
        return match.group(1).replace(',', '')
   
# Process each row in the 'Torque' column of jaipur_cars_df using iloc
for i in range(len(jaipur_cars_df)):
    jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Torque')] = process_numeric(jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Torque')])   

def process_numeric(value):
    text = str(value)
    match = re.search(r'([\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')

for i in range(len(jaipur_cars_df)):
    jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('km')] = process_numeric(jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('km')])
    
def process_numeric(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find decimal numbers or integers, allowing commas
    match = re.search(r'([\d,]+\.?\d*)', text)  # Capture numeric value with commas

    if match:
        # Return the numeric value, removing commas for numerical processing
        return match.group(1).replace(',', '')

# Process each row in the 'Height' column of jaipur_cars_df using iloc
for i in range(len(jaipur_cars_df)):
    jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Height')] = process_numeric(jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Height')])
  
# Function to process numeric values from string (for 'Wheel Base' column)
def process_numeric(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find decimal numbers or integers, allowing commas
    match = re.search(r'([\d,]+\.?\d*)', text)  # Capture numeric value with commas

    if match:
        # Return the numeric value, removing commas for numerical processing
        return match.group(1).replace(',', '')
 
# Process each row in the 'Wheel Base' column of jaipur_cars_df using iloc
for i in range(len(jaipur_cars_df)):
    jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Wheel Base')] = process_numeric(jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Wheel Base')])

# Function to process numeric values from a string (for 'Kerb Weight' column)
def process_numeric(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find decimal numbers or integers, allowing commas
    match = re.search(r'([\d,]+\.?\d*)', text)  # Capture numeric value with commas

    if match:
        # Return the numeric value, removing commas for numerical processing
        return match.group(1).replace(',', '')
  
# Process each row in the 'Kerb Weight' column of jaipur_cars_df using iloc
for i in range(len(jaipur_cars_df)):
    jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Kerb Weight')] = process_numeric(jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Kerb Weight')])


# Function to process numeric values from a string (for 'Gear Box' column)
def process_numeric(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find decimal numbers or integers, allowing commas
    match = re.search(r'([\d,]+\.?\d*)', text)  # Capture numeric value with commas

    if match:
        # Return the numeric value, removing commas for numerical processing
        return match.group(1).replace(',', '')
  
# Process each row in the 'Gear Box' column of jaipur_cars_df using iloc
for i in range(len(jaipur_cars_df)):
    jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Gear Box')] = process_numeric(jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Gear Box')])

def process_cargo_volume(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find decimal numbers or integers, allowing commas
    match = re.search(r'([\d,]+\.?\d*)', text)  # Capture numeric value with commas

    if match:
        # Return the numeric cargo volume value, removing commas for numerical processing
        return match.group(1).replace(',', '')
  

# Process each row in the 'Cargo Volume' column of jaipur_cars_df using iloc
for i in range(len(jaipur_cars_df)):
    # Use iloc to avoid KeyError for indices that may not be continuous
    jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Cargo Volumn')] = process_cargo_volume(jaipur_cars_df.iloc[i, jaipur_cars_df.columns.get_loc('Cargo Volumn')])


# Replace 'bt' and 'Cargo Volumn' with actual column names if they differ
group_column = 'bt'
target_column = 'Cargo Volumn'

# Convert the target column to numeric, coercing errors to NaN
jaipur_cars_df[target_column] = pd.to_numeric(jaipur_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

def clean_weight(value):
    if pd.isnull(value):
        return None
    value = re.sub(r'[^0-9,]', '', str(value))  # Keep numbers and commas
    if ',' in value:
        value = value.split(',')[-1]  # Keep the value after the comma
    return int(value) if value.isdigit() else None

# Apply the clean_weight function to the respective columns in jaipur_cars_df
jaipur_cars_df['Kerb Weight'] = jaipur_cars_df['Kerb Weight'].apply(clean_weight)
jaipur_cars_df['Height'] = jaipur_cars_df['Height'].apply(clean_weight)


jaipur_cars_df[['km', 'price']] = jaipur_cars_df[['km', 'price']].apply(lambda x: x.astype(float))
jaipur_cars_df['Seating Capacity'] = jaipur_cars_df['Seating Capacity'].astype(float)

jaipur_cars_df.drop(columns=['trendingText.imgUrl', 'trendingText.heading', 'trendingText.desc'], inplace=True)


group_column = 'bt'
target_column = 'Super Charger'

# Perform group-wise mode imputation
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Most common value'))

group_column = 'bt'
target_column = 'Drive Type'

# Perform group-wise mode imputation
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Most common value'))

group_column = 'bt'
target_column = 'Fuel Suppy System'

# Perform group-wise mode imputation
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Most common value'))

group_column = 'bt'
target_column = 'Turbo Charger'

# Perform group-wise mode imputation
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Most common value'))

group_column = 'bt'
target_column = 'Engine Type'

# Perform group-wise mode imputation
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Most common value'))

group_column = 'bt'
target_column = 'Gear Box'

# Convert the target column to numeric, coercing errors to NaN
jaipur_cars_df[target_column] = pd.to_numeric(jaipur_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(np.ceil(x.mean())))


group_column = 'bt'
target_column = 'Steering Type'

# Perform group-wise mode imputation
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Most common value'))

group_column = 'bt'
target_column = 'Wheel Base'

# Convert the target column to numeric, coercing errors to NaN
jaipur_cars_df[target_column] = pd.to_numeric(jaipur_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_column = 'Cargo Volumn'
# Convert the target column 'Cargo Volume' to numeric, coercing errors to NaN
jaipur_cars_df['Cargo Volumn'] = pd.to_numeric(jaipur_cars_df['Cargo Volumn'], errors='coerce')

# Perform group-wise mean imputation for 'Cargo Volume'
jaipur_cars_df['Cargo Volumn'] = jaipur_cars_df.groupby(group_column)['Cargo Volumn'].transform(lambda x: x.fillna(x.mean()))


group_column = 'bt'
target_column = 'Mileage'

# Convert the target column to numeric, coercing errors to NaN
jaipur_cars_df[target_column] = pd.to_numeric(jaipur_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_column = 'Height'

# Convert the target column to numeric, coercing errors to NaN
jaipur_cars_df[target_column] = pd.to_numeric(jaipur_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_column = 'Kerb Weight'

# Convert the target column to numeric, coercing errors to NaN
jaipur_cars_df[target_column] = pd.to_numeric(jaipur_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_column_front_brake = 'Front Brake Type'

jaipur_cars_df['Front Brake Type'] = jaipur_cars_df.groupby('bt')['Front Brake Type'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Most common value'))

group_column = 'bt'
target_column_front_brake = 'Rear Brake Type'

jaipur_cars_df['Rear Brake Type'] = jaipur_cars_df.groupby('bt')['Rear Brake Type'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Most common value'))

group_column = 'bt'
target_column = 'Torque'

# Convert the target column to numeric, coercing errors to NaN
jaipur_cars_df[target_column] = pd.to_numeric(jaipur_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_column = 'Registration Year'

# Convert the target column to numeric, coercing errors to NaN
jaipur_cars_df[target_column] = pd.to_numeric(jaipur_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation and round up the mean
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(np.ceil(x.mean())))

group_column = 'bt'
target_column = 'Value Configuration'

# Perform group-wise mode imputation
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Most common value'))

group_column = 'bt'
target_column = 'RTO'
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Most common value'))

group_column = 'bt'
target_column = 'No of Cylinder'

# Convert the target column to numeric, ignoring non-numeric entries and coercing errors to NaN
jaipur_cars_df[target_column] = pd.to_numeric(jaipur_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation and round the mean to an integer
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(round(x.mean())))

group_column = 'bt'
target_column = 'Values per Cylinder'

# Convert the target column to numeric, ignoring non-numeric entries and coercing errors to NaN
jaipur_cars_df[target_column] = pd.to_numeric(jaipur_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation and round the mean to an integer
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(round(x.mean())))

group_column = 'bt'
target_column = 'Year of Manufacture'

# Convert the target column to numeric, ignoring non-numeric entries and coercing errors to NaN
jaipur_cars_df[target_column] = pd.to_numeric(jaipur_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation and round the mean to an integer
jaipur_cars_df[target_column] = jaipur_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(np.round(x.mean())))

# Replace "Not available" with NaN across all columns
jaipur_cars_df.replace("Not available", np.nan, inplace=True)

jaipur_cars_df = jaipur_cars_df.dropna()
jaipur_cars_df = jaipur_cars_df.drop(columns=['Transmission'])

#perform one hot encoding
jaipur_cars_df = pd.get_dummies(jaipur_cars_df, columns=['ft'])
columns_to_convert = ['ft_Cng', 'ft_Diesel', 'ft_Electric', 'ft_Lpg', 'ft_Petrol']
jaipur_cars_df[columns_to_convert] = jaipur_cars_df[columns_to_convert].astype(int)

jaipur_cars_df = pd.get_dummies(jaipur_cars_df, columns=['bt'], prefix='bt')
bt_columns = ['bt_Coupe', 'bt_Hatchback', 'bt_MUV', 'bt_Minivans', 'bt_SUV', 'bt_Sedan']
jaipur_cars_df[bt_columns] = jaipur_cars_df[bt_columns].apply(lambda x: x.astype(int))

jaipur_cars_df = pd.get_dummies(jaipur_cars_df, columns=['transmission'], prefix='transmission')

# Ensure the one-hot encoded columns for 'transmission' are converted to integers
transmission_columns = ['transmission_Automatic', 'transmission_Manual']
jaipur_cars_df[transmission_columns] = jaipur_cars_df[transmission_columns].astype(int)

# Initialize the LabelEncoder
le = LabelEncoder()

# Apply Label Encoding to specific columns in 'jaipur_cars_df'
columns_to_encode = [
    'oem', 'model', 'variantName', 'RTO', 'Color', 'Engine Type', 
    'Value Configuration', 'Fuel Suppy System', 'Insurance Validity', 
    'Turbo Charger', 'Super Charger', 'Drive Type', 'Steering Type',"Front Brake Type","Rear Brake Type"
]


# Apply Label Encoding to each column in the list
for col in columns_to_encode:
    jaipur_cars_df[col] = le.fit_transform(jaipur_cars_df[col])
    
# Initialize the Min-Max Scaler
scaler = MinMaxScaler()

# List of all numerical columns for Min-Max scaling
numerical_columns = [
    'price', 'ft_Cng', 'ft_Diesel', 'ft_Electric', 'ft_Lpg', 'ft_Petrol', 
    'Registration Year', 'km', 'Engine', 'Year of Manufacture', 
    'Height', 'Kerb Weight', 'Seating Capacity', 'Cargo Volumn', 
    'bt_Coupe', 'bt_Hatchback', 'bt_MUV', 'bt_Minivans',"ownerNo","Mileage","Torque","Values per Cylinder",
    'bt_SUV', 'bt_Sedan', 'transmission_Automatic', 'transmission_Manual',"Wheel Base", 
    'oem', 'model', 'modelYear', 'centralVariantId', 'variantName', 
    'price', 'Insurance Validity', 'RTO', 'Color', 'Engine Type', 'No of Cylinder', 
    'Value Configuration', 'Fuel Suppy System', 'Turbo Charger', 'Super Charger', 
    'Gear Box', 'Drive Type', 'Steering Type', 'Front Brake Type', 'Rear Brake Type'
]

# Apply Min-Max scaling to the specified numerical columns
jaipur_cars_df[numerical_columns] = scaler.fit_transform(jaipur_cars_df[numerical_columns])

output_file_path = 'C:/Users/bhavan/Downloads/jaipur_cars_df.csv'
jaipur_cars_df.to_csv(output_file_path, index=False)



#moving to the next city. have to perform mean and one heart encoding in the jaipur_cars_df  and then move to the next city to perform


file_path = 'C:/Users/bhavan/Downloads/hyderabad_cars.xlsx'
hyderabad_cars_df = pd.read_excel(file_path)

# Convert 'new_car_detail' to dictionary values
hyderabad_cars_df['new_car_detail'] = hyderabad_cars_df['new_car_detail'].apply(eval)

# Normalize the dictionary values into separate columns
details_df = pd.json_normalize(hyderabad_cars_df['new_car_detail'])

# Concatenate the new columns with the original DataFrame
hyderabad_cars_df = pd.concat([hyderabad_cars_df.drop(columns=['new_car_detail']), details_df], axis=1)

hyderabad_cars_df = hyderabad_cars_df.drop(columns=['priceActual', 'priceSaving', 'priceFixedText'])

hyderabad_cars_df = hyderabad_cars_df.drop(columns=['owner'])

def extract_all_numbers(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find all numeric values in the text
    matches = re.findall(r'[\d,]+\.?\d*', text)  # Capture all numeric values, allowing for commas

    # Join the found matches into a single string
    if matches:
        # Remove commas and join numbers without brackets
        return ', '.join(match.replace(',', '') for match in matches)  # Join with a comma
    else:
        return "Not available"  # Return 'Not available' if no numbers are found

# Process each row in the 'price' column of hyderabad_cars_df
for i in range(len(hyderabad_cars_df)):
    hyderabad_cars_df.at[i, 'price'] = extract_all_numbers(hyderabad_cars_df.at[i, 'price'])
    

# Function to convert string to dictionary
def str_to_dict(row):
    return ast.literal_eval(row)

# Apply the function to the Series in 'new_car_overview' column
data_dicts = hyderabad_cars_df["new_car_overview"].apply(str_to_dict)

# Extract the 'top' data from each dictionary and create a DataFrame
data_list = [{item['key']: item['value'] for item in d['top']} for d in data_dicts]

# Convert the list of dictionaries into a DataFrame
df_overview = pd.DataFrame(data_list)

# Concatenate the new columns with the original DataFrame
hyderabad_cars_df = pd.concat([hyderabad_cars_df, df_overview], axis=1)

# Function to extract the year from the 'Registration Year' column
def extract_year(value):
    if pd.isna(value):
        return np.nan
    try:
        return int(value)
    except ValueError:
        # Extract the year from strings like 'Jul 2017'
        return int(value.split()[-1])

# Apply the function to the 'Registration Year' column in hyderabad_cars_df
hyderabad_cars_df['Registration Year'] = hyderabad_cars_df['Registration Year'].apply(extract_year)

def str_to_dict(row):
    return ast.literal_eval(row)

data_dicts_specs = hyderabad_cars_df["new_car_specs"].apply(str_to_dict)
data_list_specs = []
for d in data_dicts_specs:
    top_dict = {item['key']: item['value'] for item in d['top']}
    for section in d['data']:
        section_dict = {item['key']: item['value'] for item in section['list']}
        top_dict.update(section_dict)
    data_list_specs.append(top_dict)

df_specs = pd.DataFrame(data_list_specs)

hyderabad_cars_df = pd.concat([hyderabad_cars_df, df_specs], axis=1)

hyderabad_cars_df = hyderabad_cars_df.drop(columns=[col for col in ['Ground Clearance Unladen', 'Compression Ratio', 'BoreX Stroke', 'Alloy Wheel Size', 'Turning Radius', 'Tyre Type','Wheel Size','Gross Weight',"Top Speed","Acceleration","Rear Tread","Front Tread","Kms Driven",'Width','Length',"Max Power","Max Torque","No Door Numbers"] if col in hyderabad_cars_df.columns])
hyderabad_cars_df = hyderabad_cars_df.drop('Values per Cylinder', axis=1)
hyderabad_cars_df = hyderabad_cars_df.drop(['new_car_overview', 'new_car_feature', 'new_car_specs', 'car_links'], axis=1)


# Replace "Not available" with NaN across all columns in 'hyderabad_cars_df'
hyderabad_cars_df.replace("Not available", np.nan, inplace=True)


# Function to process numeric values from a string (for all specified columns)
def process_numeric(value):
    # Convert non-string values to strings to enable regex processing
    text = str(value)

    # Use regex to find decimal numbers or integers, allowing commas
    match = re.search(r'([\d,]+\.?\d*)', text)  # Capture numeric value with commas

    if match:
        # Get the numeric value and remove commas
        processed_value = match.group(1).replace(',', '')
        
        # Ensure the processed value is not empty before converting
        if processed_value:
            return float(processed_value) if '.' in processed_value else int(processed_value)
    
    return None  # Return None if no numeric value is found or processed value is empty


# List of columns to process
columns_to_process = ['Engine', 'Mileage', 'Torque', 'Height', 'Wheel Base', 'Kerb Weight', 'Gear Box',"km",'Cargo Volumn']


# Process each column in the 'columns_to_process' list for hyderabad_cars_df
for column in columns_to_process:
    for i in range(len(hyderabad_cars_df)):
        # Use iloc to avoid KeyError for indices that may not be continuous
        hyderabad_cars_df.iloc[i, hyderabad_cars_df.columns.get_loc(column)] = process_numeric(hyderabad_cars_df.iloc[i, hyderabad_cars_df.columns.get_loc(column)])

def clean_weight(value):
    if pd.isnull(value):
        return None
    value = re.sub(r'[^0-9,]', '', str(value))  # Keep numbers and commas
    if ',' in value:
        value = value.split(',')[-1]  # Keep the value after the comma
    return int(value) if value.isdigit() else None

hyderabad_cars_df['Kerb Weight'] = hyderabad_cars_df['Kerb Weight'].apply(clean_weight)
hyderabad_cars_df['Height'] = hyderabad_cars_df['Height'].apply(clean_weight)
hyderabad_cars_df['Cargo Volumn'] = hyderabad_cars_df['Cargo Volumn'].apply(clean_weight)


# Remove rows where 'bt' column is NaN or contains only whitespace
hyderabad_cars_df = hyderabad_cars_df[hyderabad_cars_df['bt'].str.strip().ne('')]

group_column = 'bt'
target_column = 'Value Configuration'

# Perform group-wise mode imputation and use the mode to fill NaN values
hyderabad_cars_df[target_column] = hyderabad_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_columns = 'Turbo Charger'
hyderabad_cars_df['Turbo Charger'] = hyderabad_cars_df.groupby('bt')['Turbo Charger'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_columns = 'Super Charger'
hyderabad_cars_df['Super Charger'] = hyderabad_cars_df.groupby('bt')['Super Charger'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_columns = 'Fuel Suppy System'
hyderabad_cars_df['Fuel Suppy System'] = hyderabad_cars_df.groupby('bt')['Fuel Suppy System'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))


group_column = 'bt'
target_columns = 'Drive Type'
hyderabad_cars_df['Drive Type'] = hyderabad_cars_df.groupby('bt')['Drive Type'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_column = 'Cargo Volumn'

# Convert the target column to numeric, ignoring non-numeric entries and coercing errors to NaN
hyderabad_cars_df[target_column] = pd.to_numeric(hyderabad_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation and round the mean to an integer
hyderabad_cars_df[target_column] = hyderabad_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_columns = ['Gear Box', 'Year of Manufacture', 'Registration Year']

# Convert columns to numeric and apply group-wise mean imputation with rounding
hyderabad_cars_df[target_columns] = hyderabad_cars_df[target_columns].apply(pd.to_numeric, errors='coerce')
hyderabad_cars_df[target_columns] = hyderabad_cars_df.groupby(group_column)[target_columns].transform(lambda x: x.fillna(round(x.mean())))


group_column = 'bt'
target_columns = 'Engine Type'
hyderabad_cars_df['Engine Type'] = hyderabad_cars_df.groupby('bt')['Engine Type'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_columns = 'Steering Type'
hyderabad_cars_df['Steering Type'] = hyderabad_cars_df.groupby('bt')['Steering Type'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_column = 'Mileage'

# Convert the target column to numeric, ignoring non-numeric entries and coercing errors to NaN
hyderabad_cars_df[target_column] = pd.to_numeric(hyderabad_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation without rounding
hyderabad_cars_df[target_column] = hyderabad_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_column = 'Wheel Base'

# Convert the target column to numeric, ignoring non-numeric entries and coercing errors to NaN
hyderabad_cars_df[target_column] = pd.to_numeric(hyderabad_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation without rounding
hyderabad_cars_df[target_column] = hyderabad_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_columns = ['Front Brake Type', 'RTO']
hyderabad_cars_df[['Front Brake Type', 'RTO']] = hyderabad_cars_df.groupby('bt')[['Front Brake Type', 'RTO']].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_column = 'Rear Brake Type'
hyderabad_cars_df['Rear Brake Type'] = hyderabad_cars_df.groupby('bt')['Rear Brake Type'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_column = 'Height'
hyderabad_cars_df[target_column] = pd.to_numeric(hyderabad_cars_df[target_column], errors='coerce')
hyderabad_cars_df[target_column] = hyderabad_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_column = 'Torque'

# Convert the target column to numeric, ignoring non-numeric entries and coercing errors to NaN
hyderabad_cars_df[target_column] = pd.to_numeric(hyderabad_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation without rounding
hyderabad_cars_df[target_column] = hyderabad_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_column = 'Kerb Weight'

# Convert the target column to numeric, ignoring non-numeric entries and coercing errors to NaN
hyderabad_cars_df[target_column] = pd.to_numeric(hyderabad_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation without rounding
hyderabad_cars_df[target_column] = hyderabad_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_column = 'No of Cylinder'

# Convert the target column to numeric, ignoring non-numeric entries and coercing errors to NaN
hyderabad_cars_df[target_column] = pd.to_numeric(hyderabad_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation without rounding
hyderabad_cars_df[target_column] = hyderabad_cars_df.groupby(group_column)[target_column].transform(lambda x: np.ceil(x.fillna(x.mean())))

columns_to_check = ['Cargo Volumn', 'Mileage', 'Super Charger', 'Fuel Suppy System', 'Value Configuration', 'Engine Displacement', 'Color', 'Engine', 'Displacement']
hyderabad_cars_df = hyderabad_cars_df.dropna(subset=columns_to_check)

# Convert specific columns to float
hyderabad_cars_df['km'] = pd.to_numeric(hyderabad_cars_df['km'], errors='coerce')
hyderabad_cars_df['price'] = pd.to_numeric(hyderabad_cars_df['price'], errors='coerce')
hyderabad_cars_df['Seating Capacity'] = pd.to_numeric(hyderabad_cars_df['Seating Capacity'], errors='coerce')
hyderabad_cars_df['Engine Displacement'] = pd.to_numeric(hyderabad_cars_df['Engine Displacement'], errors='coerce')

# Drop the specified columns including 'Seats'
hyderabad_cars_df = hyderabad_cars_df.drop(['trendingText.imgUrl', 'trendingText.heading', 'trendingText.desc', 'Seats',"Engine"], axis=1)
hyderabad_cars_df = hyderabad_cars_df.drop(columns=['Fuel Type'])
hyderabad_cars_df = hyderabad_cars_df.drop(columns=['Engine Displacement'])
hyderabad_cars_df = hyderabad_cars_df.drop(columns=['Transmission'])


#perform one hot encoding
hyderabad_cars_df = pd.get_dummies(hyderabad_cars_df, columns=['ft'])
columns_to_convert = ['ft_Cng', 'ft_Diesel', 'ft_Electric', 'ft_Lpg', 'ft_Petrol']
hyderabad_cars_df[columns_to_convert] = hyderabad_cars_df[columns_to_convert].astype(int)


hyderabad_cars_df = pd.get_dummies(hyderabad_cars_df, columns=['bt'], prefix='bt')
bt_columns = ['bt_Hatchback', 'bt_MUV', 'bt_Minivans', 'bt_SUV', 'bt_Sedan']
hyderabad_cars_df[bt_columns] = hyderabad_cars_df[bt_columns].apply(lambda x: x.astype(int))

hyderabad_cars_df = pd.get_dummies(hyderabad_cars_df, columns=['transmission'], prefix='transmission')

transmission_columns = ['transmission_Automatic', 'transmission_Manual']
hyderabad_cars_df[transmission_columns] = hyderabad_cars_df[transmission_columns].astype(int)

# Initialize the LabelEncoder
le = LabelEncoder()

# Apply Label Encoding to specific columns in 'hyderabad_cars_df'
columns_to_encode = [
    'oem', 'model', 'variantName', 'RTO', 'Color', 'Engine Type',"Ownership",
    'Value Configuration', 'Fuel Suppy System', 'Insurance Validity',
    'Turbo Charger', 'Super Charger', 'Drive Type', 'Steering Type',
    "Front Brake Type", "Rear Brake Type"
]

# Apply Label Encoding to each column in the list
for col in columns_to_encode:
    hyderabad_cars_df[col] = le.fit_transform(hyderabad_cars_df[col])
      
# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# List of all numerical columns for Min-Max scaling
numerical_columns = [
    'price', 'ft_Cng', 'ft_Diesel', 'ft_Electric', 'ft_Lpg', 'ft_Petrol', 
    'Registration Year', 'km', 'Displacement', 'Year of Manufacture', 
    'Height', 'Kerb Weight', 'Seating Capacity', 'Cargo Volumn', 
    'bt_Hatchback', 'bt_MUV', 'bt_Minivans',"Ownership","Mileage","Torque",
    'bt_SUV', 'bt_Sedan', 'transmission_Automatic', 'transmission_Manual',"Wheel Base", 
    'oem', 'model', 'modelYear', 'centralVariantId', 'variantName', 
    'price', 'Insurance Validity', 'RTO', 'Color', 'Engine Type', 'No of Cylinder', 
    'Value Configuration', 'Fuel Suppy System', 'Turbo Charger', 'Super Charger', 
    'Gear Box', 'Drive Type', 'Steering Type', 'Front Brake Type', 'Rear Brake Type'
]

# Apply Min-Max scaling to the specified numerical columns
hyderabad_cars_df[numerical_columns] = scaler.fit_transform(hyderabad_cars_df[numerical_columns])



output_file_path = 'C:/Users/bhavan/Downloads/hyderabad_cars_df.csv'
hyderabad_cars_df.to_csv(output_file_path, index=False)


#moving to the next city

file_path_delhi = 'C:/Users/bhavan/Downloads/delhi_cars.xlsx'
delhi_cars_df = pd.read_excel(file_path_delhi)

# Convert 'new_car_detail' to dictionary values
delhi_cars_df['new_car_detail'] = delhi_cars_df['new_car_detail'].apply(eval)

# Normalize the dictionary values into separate columns
details_df = pd.json_normalize(delhi_cars_df['new_car_detail'])

# Concatenate the new columns with the original DataFrame
delhi_cars_df = pd.concat([delhi_cars_df.drop(columns=['new_car_detail']), details_df], axis=1)

delhi_cars_df = delhi_cars_df.drop(columns=['priceActual', 'priceSaving', 'priceFixedText'])

delhi_cars_df = delhi_cars_df.drop(columns=['owner'])

def extract_all_numbers(value):
    text = str(value)
    matches = re.findall(r'[\d,]+\.?\d*', text)
    return ', '.join(match.replace(',', '') for match in matches) if matches else "Not available"

for i in range(len(delhi_cars_df)):
    delhi_cars_df.at[i, 'price'] = extract_all_numbers(delhi_cars_df.at[i, 'price'])
    

def str_to_dict(row):
    return ast.literal_eval(row)

data_dicts = delhi_cars_df["new_car_overview"].apply(str_to_dict)
data_list = [{item['key']: item['value'] for item in d['top']} for d in data_dicts]
df_overview = pd.DataFrame(data_list)
delhi_cars_df = pd.concat([delhi_cars_df, df_overview], axis=1)

def extract_year(value):
    if pd.isna(value):
        return np.nan
    try:
        return int(value)
    except ValueError:
        return int(value.split()[-1])

delhi_cars_df['Registration Year'] = delhi_cars_df['Registration Year'].apply(extract_year)

def str_to_dict(row):
    return ast.literal_eval(row)

data_dicts_specs = delhi_cars_df["new_car_specs"].apply(str_to_dict)
data_list_specs = []
for d in data_dicts_specs:
    top_dict = {item['key']: item['value'] for item in d['top']}
    for section in d['data']:
        section_dict = {item['key']: item['value'] for item in section['list']}
        top_dict.update(section_dict)
    data_list_specs.append(top_dict)

df_specs = pd.DataFrame(data_list_specs)
delhi_cars_df = pd.concat([delhi_cars_df, df_specs], axis=1)

delhi_cars_df = delhi_cars_df.drop(columns=[col for col in ['Ground Clearance Unladen', 'Compression Ratio', 'BoreX Stroke', 'Alloy Wheel Size', 'Turning Radius', 'Tyre Type', 'Wheel Size', 'Gross Weight', "Top Speed", "Acceleration", "Rear Tread", "Front Tread", "Kms Driven", 'Width', 'Length', "Max Power", "Max Torque", "No Door Numbers","Ownership","Engine","Displacement","Seats"] if col in delhi_cars_df.columns])

def process_numeric(value):
    text = str(value)
    match = re.search(r'([\d,]+\.?\d*)', text)
    if match:
        processed_value = match.group(1).replace(',', '')
        if processed_value:
            return float(processed_value) if '.' in processed_value else int(processed_value)
    return None

columns_to_process = ["Engine Displacement",'Mileage', 'Torque', 'Height', 'Wheel Base', 'Kerb Weight', 'Gear Box', 'Cargo Volumn']

for column in columns_to_process:
    for i in range(len(delhi_cars_df)):
        delhi_cars_df.iloc[i, delhi_cars_df.columns.get_loc(column)] = process_numeric(delhi_cars_df.iloc[i, delhi_cars_df.columns.get_loc(column)])

def clean_weight(value):
    if pd.isnull(value):
        return None
    value = re.sub(r'[^0-9]', '', str(value))  # Remove anything that is not a digit
    return int(value) if value.isdigit() else None

delhi_cars_df['km'] = delhi_cars_df['km'].apply(clean_weight)


group_column = 'bt'
target_column = 'Value Configuration'

# Perform group-wise mode imputation and use the mode to fill NaN values
delhi_cars_df[target_column] = delhi_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_column = 'Super Charger'

delhi_cars_df[target_column] = delhi_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_column = 'Fuel Suppy System'

delhi_cars_df[target_column] = delhi_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_column = 'Cargo Volumn'

# Convert the target column to numeric, coercing errors to NaN
delhi_cars_df[target_column] = pd.to_numeric(delhi_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation
delhi_cars_df[target_column] = delhi_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_column = 'Drive Type'

delhi_cars_df[target_column] = delhi_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_column = 'Turbo Charger'

delhi_cars_df[target_column] = delhi_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))



group_column = 'bt'
target_column = 'Mileage'

# Convert the target column to numeric, coercing errors to NaN
delhi_cars_df[target_column] = pd.to_numeric(delhi_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation
delhi_cars_df[target_column] = delhi_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_column = 'Steering Type'

delhi_cars_df[target_column] = delhi_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_column = 'Gear Box'

# Convert the target column to numeric, coercing errors to NaN
delhi_cars_df[target_column] = pd.to_numeric(delhi_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation and round up
delhi_cars_df[target_column] = delhi_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()).round())

group_column = 'bt'
target_column = 'Engine Type'

delhi_cars_df[target_column] = delhi_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_column = 'Wheel Base'

# Convert the target column to numeric, coercing errors to NaN
delhi_cars_df[target_column] = pd.to_numeric(delhi_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation without rounding
delhi_cars_df[target_column] = delhi_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'

target_column_front_brake = 'Front Brake Type'
delhi_cars_df[target_column_front_brake] = delhi_cars_df.groupby(group_column)[target_column_front_brake].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_column_rto = 'RTO'

delhi_cars_df['RTO'] = delhi_cars_df.groupby(group_column)['RTO'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

# For 'Rear Brake Type' column
target_column_rear_brake = 'Rear Brake Type'
delhi_cars_df[target_column_rear_brake] = delhi_cars_df.groupby(group_column)[target_column_rear_brake].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'

# For 'Height' column
target_column_height = 'Height'
delhi_cars_df[target_column_height] = pd.to_numeric(delhi_cars_df[target_column_height], errors='coerce')
delhi_cars_df[target_column_height] = delhi_cars_df.groupby(group_column)[target_column_height].transform(lambda x: x.fillna(x.mean()))

# For 'Kerb Weight' column
target_column_kerb_weight = 'Kerb Weight'
delhi_cars_df[target_column_kerb_weight] = pd.to_numeric(delhi_cars_df[target_column_kerb_weight], errors='coerce')
delhi_cars_df[target_column_kerb_weight] = delhi_cars_df.groupby(group_column)[target_column_kerb_weight].transform(lambda x: x.fillna(x.mean()))

# For 'Torque' column
target_column_torque = 'Torque'
delhi_cars_df[target_column_torque] = pd.to_numeric(delhi_cars_df[target_column_torque], errors='coerce')
delhi_cars_df[target_column_torque] = delhi_cars_df.groupby(group_column)[target_column_torque].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'

group_column = 'bt'

# For 'Values per Cylinder' column
target_column_values_per_cylinder = 'Values per Cylinder'
delhi_cars_df[target_column_values_per_cylinder] = pd.to_numeric(delhi_cars_df[target_column_values_per_cylinder], errors='coerce')
delhi_cars_df[target_column_values_per_cylinder] = delhi_cars_df.groupby(group_column)[target_column_values_per_cylinder].transform(lambda x: x.fillna(x.mean()).apply(np.ceil))

group_column = 'bt'
target_column_registration_year = 'Registration Year'
delhi_cars_df[target_column_registration_year] = pd.to_numeric(delhi_cars_df[target_column_registration_year], errors='coerce')
delhi_cars_df[target_column_registration_year] = delhi_cars_df.groupby(group_column)[target_column_registration_year].transform(lambda x: x.fillna(x.mean()).apply(np.ceil))

# For 'No of Cylinder' column
target_column_no_of_cylinders = 'No of Cylinder'
delhi_cars_df[target_column_no_of_cylinders] = pd.to_numeric(delhi_cars_df[target_column_no_of_cylinders], errors='coerce')
delhi_cars_df[target_column_no_of_cylinders] = delhi_cars_df.groupby(group_column)[target_column_no_of_cylinders].transform(lambda x: x.fillna(x.mean()).apply(np.ceil))

delhi_cars_df = delhi_cars_df.dropna(subset=['Turbo Charger', 'Cargo Volumn', 'Insurance Validity', 'Fuel Suppy System', 'Mileage', 'Gear Box', 'Kerb Weight', 'Drive Type', 'Seating Capacity', 'Value Configuration', 'Super Charger'])

delhi_cars_df = delhi_cars_df.drop(columns=['new_car_overview', 'new_car_feature', 'new_car_specs', 'car_links'])
delhi_cars_df = delhi_cars_df.drop(columns=['trendingText.imgUrl', 'trendingText.heading', 'trendingText.desc'])
delhi_cars_df = delhi_cars_df.drop(columns=['Fuel Type', 'Transmission'])


#perform one hot encoding
delhi_cars_df = pd.get_dummies(delhi_cars_df, columns=['ft'])
columns_to_convert = ['ft_Cng', 'ft_Diesel', 'ft_Electric', 'ft_Lpg', 'ft_Petrol']
delhi_cars_df[columns_to_convert] = delhi_cars_df[columns_to_convert].astype(int)

delhi_cars_df = pd.get_dummies(delhi_cars_df, columns=['bt'], prefix='bt')
bt_columns = ['bt_Coupe', 'bt_Hatchback', 'bt_MUV','bt_SUV', 'bt_Sedan',"bt_Convertibles"]
delhi_cars_df[bt_columns] = delhi_cars_df[bt_columns].apply(lambda x: x.astype(int))

delhi_cars_df = pd.get_dummies(delhi_cars_df, columns=['transmission'], prefix='transmission')

# Ensure the one-hot encoded columns for 'transmission' are converted to integers
transmission_columns = ['transmission_Automatic', 'transmission_Manual']
delhi_cars_df[transmission_columns] = delhi_cars_df[transmission_columns].astype(int)

# Initialize the LabelEncoder
le = LabelEncoder()

# Apply Label Encoding to specific columns in 'delhi_cars_df'
columns_to_encode = [
    'oem', 'model', 'variantName', 'RTO', 'Color', 'Engine Type', 
    'Value Configuration', 'Fuel Suppy System', 'Insurance Validity', 
    'Turbo Charger', 'Super Charger', 'Drive Type', 'Steering Type',
    "Front Brake Type", "Rear Brake Type",
]

# Apply Label Encoding to each column in the list
for col in columns_to_encode:
    delhi_cars_df[col] = le.fit_transform(delhi_cars_df[col])
    
# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# List of all numerical columns for Min-Max scaling
numerical_columns = [
    'price', 'ft_Cng', 'ft_Diesel', 'ft_Electric', 'ft_Lpg', 'ft_Petrol', 
    'Registration Year', 'km', 'Engine Displacement', 'Year of Manufacture', 
    'Height', 'Kerb Weight', 'Seating Capacity', 'Cargo Volumn', 
    'bt_Coupe', 'bt_Hatchback', 'bt_MUV',"ownerNo","Mileage","Torque","Values per Cylinder",
    'bt_SUV', 'bt_Sedan', 'transmission_Automatic', 'transmission_Manual',"Wheel Base", 
    'oem', 'model', 'modelYear', 'centralVariantId', 'variantName', 
    'price', 'Insurance Validity', 'RTO', 'Color', 'Engine Type', 'No of Cylinder', 
    'Value Configuration', 'Fuel Suppy System', 'Turbo Charger', 'Super Charger', 
    'Gear Box', 'Drive Type', 'Steering Type', 'Front Brake Type', 'Rear Brake Type'
]

# Apply Min-Max scaling to the specified numerical columns
delhi_cars_df[numerical_columns] = scaler.fit_transform(delhi_cars_df[numerical_columns])


output_file_path = 'C:/Users/bhavan/Downloads/delhi_cars_df.csv'
delhi_cars_df.to_csv(output_file_path, index=False)


#moving to the next city

file_path_chennai = 'C:/Users/bhavan/Downloads/chennai_cars.xlsx'
chennai_cars_df = pd.read_excel(file_path_chennai)

# Convert 'new_car_detail' to dictionary values
chennai_cars_df['new_car_detail'] = chennai_cars_df['new_car_detail'].apply(eval)

# Normalize the dictionary values into separate columns
details_df = pd.json_normalize(chennai_cars_df['new_car_detail'])

# Concatenate the new columns with the original DataFrame
chennai_cars_df = pd.concat([chennai_cars_df.drop(columns=['new_car_detail']), details_df], axis=1)

# Drop unnecessary columns
chennai_cars_df = chennai_cars_df.drop(columns=['priceActual', 'priceSaving', 'priceFixedText'])

chennai_cars_df = chennai_cars_df.drop(columns=['owner'])

# Function to extract all numbers from a value
def extract_all_numbers(value):
    text = str(value)
    matches = re.findall(r'[\d,]+\.?\d*', text)
    return ', '.join(match.replace(',', '') for match in matches) if matches else ""

# Apply the extract_all_numbers function to the 'price' column
for i in range(len(chennai_cars_df)):
    chennai_cars_df.at[i, 'price'] = extract_all_numbers(chennai_cars_df.at[i, 'price'])

# Function to convert string to dictionary
def str_to_dict(row):
    return ast.literal_eval(row) if row else {}

# Apply the str_to_dict function to the 'new_car_overview' column
data_dicts = chennai_cars_df["new_car_overview"].apply(str_to_dict)

# Extract data from the dictionaries and create a new DataFrame
data_list = [{item['key']: item['value'] for item in d['top']} for d in data_dicts]
df_overview = pd.DataFrame(data_list)

# Concatenate the new DataFrame with the original
chennai_cars_df = pd.concat([chennai_cars_df, df_overview], axis=1)

# Function to extract the year from the 'Registration Year' column
def extract_year(value):
    if pd.isna(value):
        return np.nan
    try:
        return int(value)
    except ValueError:
        return int(value.split()[-1])

# Apply the extract_year function to the 'Registration Year' column
chennai_cars_df['Registration Year'] = chennai_cars_df['Registration Year'].apply(extract_year)

def str_to_dict(row):
    return ast.literal_eval(row) if row else {}

data_dicts_specs = chennai_cars_df["new_car_specs"].apply(str_to_dict)
data_list_specs = []

for d in data_dicts_specs:
    top_dict = {item['key']: item['value'] for item in d['top']}
    for section in d['data']:
        section_dict = {item['key']: item['value'] for item in section['list']}
        top_dict.update(section_dict)
    data_list_specs.append(top_dict)

df_specs = pd.DataFrame(data_list_specs)
chennai_cars_df = pd.concat([chennai_cars_df, df_specs], axis=1)

chennai_cars_df = chennai_cars_df.drop(columns=[col for col in ['Ground Clearance Unladen', 'Compression Ratio', 'BoreX Stroke', 'Alloy Wheel Size', 'Turning Radius', 'Tyre Type', 'Wheel Size', 'Gross Weight', "Top Speed", "Acceleration", "Rear Tread", "Front Tread", "Kms Driven", 'Width', 'Length', "Max Power", "Max Torque", "No Door Numbers", "Ownership", "Engine", "Displacement", "Seats"] if col in chennai_cars_df.columns])


def clean_weight(value):
    if pd.isnull(value):
        return None
    value = re.sub(r'[^0-9]', '', str(value))  # Remove anything that is not a digit
    return int(value) if value.isdigit() else None

chennai_cars_df['km'] = chennai_cars_df['km'].apply(clean_weight)

def process_numeric(value):
    text = str(value)
    match = re.search(r'([\d,]+\.?\d*)', text)
    if match:
        processed_value = match.group(1).replace(',', '')
        if processed_value:
            return float(processed_value) if '.' in processed_value else int(processed_value)
    return None

columns_to_process = ["Engine Displacement", 'Mileage', 'Torque', 'Height', 'Wheel Base', 'Kerb Weight', 'Gear Box', 'Cargo Volumn']

for column in columns_to_process:
    for i in range(len(chennai_cars_df)):
        chennai_cars_df.iloc[i, chennai_cars_df.columns.get_loc(column)] = process_numeric(chennai_cars_df.iloc[i, chennai_cars_df.columns.get_loc(column)])


group_column = 'bt'
target_columns = ['Value Configuration', 'Super Charger', 'Fuel Suppy System']

for target_column in target_columns:
    chennai_cars_df[target_column] = chennai_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_columns = ['Drive Type', 'Turbo Charger']

for target_column in target_columns:
    chennai_cars_df[target_column] = chennai_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_column = 'Cargo Volumn'

# Convert the target column to numeric, coercing errors to NaN
chennai_cars_df[target_column] = pd.to_numeric(chennai_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation
chennai_cars_df[target_column] = chennai_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_column = 'Gear Box'

# Convert the target column to numeric, coercing errors to NaN
chennai_cars_df[target_column] = pd.to_numeric(chennai_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation and round up
chennai_cars_df[target_column] = chennai_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()).round())

group_column = 'bt'
target_column = 'Engine Type'

chennai_cars_df[target_column] = chennai_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_column = 'RTO'

chennai_cars_df[target_column] = chennai_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_columns = ['Mileage', 'Wheel Base']

for target_column in target_columns:
    # Convert the target column to numeric, coercing errors to NaN
    chennai_cars_df[target_column] = pd.to_numeric(chennai_cars_df[target_column], errors='coerce')
    
    # Perform group-wise mean imputation
    chennai_cars_df[target_column] = chennai_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_columns = ['Steering Type', 'Front Brake Type', 'Rear Brake Type']

for target_column in target_columns:
    chennai_cars_df[target_column] = chennai_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))


group_column = 'bt'
target_columns = ['Registration Year', 'Values per Cylinder', 'No of Cylinder']

for target_column in target_columns:
    # Convert the target column to numeric, coercing errors to NaN
    chennai_cars_df[target_column] = pd.to_numeric(chennai_cars_df[target_column], errors='coerce')
    
    # Perform group-wise mean imputation and round up
    chennai_cars_df[target_column] = chennai_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()).apply(np.ceil))

group_column = 'bt'
target_columns = ['Height', 'Kerb Weight', 'Torque']

for target_column in target_columns:
    # Convert the target column to numeric, coercing errors to NaN
    chennai_cars_df[target_column] = pd.to_numeric(chennai_cars_df[target_column], errors='coerce')
    
    # Perform group-wise mean imputation
    chennai_cars_df[target_column] = chennai_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

chennai_cars_df = chennai_cars_df.dropna(subset=['Fuel Suppy System', 'Value Configuration'])
chennai_cars_df = chennai_cars_df.drop(columns=['new_car_overview', 'new_car_feature', 'new_car_specs', 'car_links'])
chennai_cars_df = chennai_cars_df.drop(columns=['trendingText.imgUrl', 'trendingText.heading', 'trendingText.desc'])



output_file_path = 'C:/Users/bhavan/Downloads/chennai_cars_df.csv'
chennai_cars_df.to_csv(output_file_path, index=False)

# moving to the next city


file_path_bangalore = 'C:/Users/bhavan/Downloads/bangalore_cars.xlsx'
bangalore_cars_df = pd.read_excel(file_path_bangalore)

# Convert 'new_car_detail' to dictionary values
bangalore_cars_df['new_car_detail'] = bangalore_cars_df['new_car_detail'].apply(eval)

# Normalize the dictionary values into separate columns
details_df = pd.json_normalize(bangalore_cars_df['new_car_detail'])

# Concatenate the new columns with the original DataFrame
bangalore_cars_df = pd.concat([bangalore_cars_df.drop(columns=['new_car_detail']), details_df], axis=1)

# Drop unnecessary columns
bangalore_cars_df = bangalore_cars_df.drop(columns=['priceActual', 'priceSaving', 'priceFixedText'])

import re

# Function to extract all numbers from a value
def extract_all_numbers(value):
    text = str(value)
    matches = re.findall(r'[\d,]+\.?\d*', text)
    return ', '.join(match.replace(',', '') for match in matches) if matches else ""

# Apply the extract_all_numbers function to the 'price' column
for i in range(len(bangalore_cars_df)):
    bangalore_cars_df.at[i, 'price'] = extract_all_numbers(bangalore_cars_df.at[i, 'price'])


# Function to convert string to dictionary
def str_to_dict(row):
    return ast.literal_eval(row) if row else {}

# Apply the str_to_dict function to the 'new_car_overview' column
data_dicts = bangalore_cars_df["new_car_overview"].apply(str_to_dict)

# Extract data from the dictionaries and create a new DataFrame
data_list = [{item['key']: item['value'] for item in d['top']} for d in data_dicts]
df_overview = pd.DataFrame(data_list)

# Concatenate the new DataFrame with the original
bangalore_cars_df = pd.concat([bangalore_cars_df, df_overview], axis=1)

# Function to extract the year from the 'Registration Year' column
def extract_year(value):
    if pd.isna(value):
        return np.nan
    try:
        return int(value)
    except ValueError:
        return int(value.split()[-1])

# Apply the extract_year function to the 'Registration Year' column
bangalore_cars_df['Registration Year'] = bangalore_cars_df['Registration Year'].apply(extract_year)

import ast

# Function to convert string to dictionary
def str_to_dict(row):
    return ast.literal_eval(row) if row else {}

# Apply the str_to_dict function to the 'new_car_specs' column
data_dicts_specs = bangalore_cars_df["new_car_specs"].apply(str_to_dict)
data_list_specs = []

# Process each dictionary and extract data
for d in data_dicts_specs:
    top_dict = {item['key']: item['value'] for item in d['top']}
    for section in d['data']:
        section_dict = {item['key']: item['value'] for item in section['list']}
        top_dict.update(section_dict)
    data_list_specs.append(top_dict)

# Create a DataFrame from the extracted data
df_specs = pd.DataFrame(data_list_specs)

# Concatenate the new DataFrame with the original
bangalore_cars_df = pd.concat([bangalore_cars_df, df_specs], axis=1)

bangalore_cars_df = bangalore_cars_df.drop(columns=[col for col in ['Ground Clearance Unladen',"owner",'Compression Ratio', 'BoreX Stroke', 'Alloy Wheel Size', 'Turning Radius', 'Tyre Type', 'Wheel Size','Gross Weight',"Top Speed","Acceleration","Rear Tread", "Front Tread", "Kms Driven", 'Width', 'Length', "Max Power", "Max Torque", "No Door Numbers", "Ownership", "Engine", "Displacement", "Seats"] if col in bangalore_cars_df.columns])

import re

# Function to process numeric values
def process_numeric(value):
    text = str(value)
    match = re.search(r'([\d,]+\.?\d*)', text)
    if match:
        processed_value = match.group(1).replace(',', '')
        if processed_value:
            return float(processed_value) if '.' in processed_value else int(processed_value)
    return None

# List of columns to process
columns_to_process = ["Engine Displacement", 'Mileage', 'Torque', 'Height', 'Wheel Base', 'Kerb Weight', 'Gear Box','Cargo Volumn']

# Apply the process_numeric function to the specified columns
for column in columns_to_process:
    for i in range(len(bangalore_cars_df)):
        bangalore_cars_df.iloc[i, bangalore_cars_df.columns.get_loc(column)] = process_numeric(bangalore_cars_df.iloc[i, bangalore_cars_df.columns.get_loc(column)])


group_column = 'bt'
target_columns = ['Value Configuration', 'Super Charger', 'Fuel Suppy System']

for target_column in target_columns:
    bangalore_cars_df[target_column] = bangalore_cars_df.groupby(group_column)[target_column].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0])
    )

group_column = 'bt'
target_columns = ['Drive Type', 'Turbo Charger']

for target_column in target_columns:
    bangalore_cars_df[target_column] = bangalore_cars_df.groupby(group_column)[target_column].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0])
    )

group_column = 'bt'
target_column = 'Cargo Volumn'

# Convert the target column to numeric, coercing errors to NaN
bangalore_cars_df[target_column] = pd.to_numeric(bangalore_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation
bangalore_cars_df[target_column] = bangalore_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_column = 'Gear Box'

# Convert the target column to numeric, coercing errors to NaN
bangalore_cars_df[target_column] = pd.to_numeric(bangalore_cars_df[target_column], errors='coerce')

# Perform group-wise mean imputation and round up
bangalore_cars_df[target_column] = bangalore_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()).round())

group_column = 'bt'
target_columns = ['Engine Type', 'RTO']

for target_column in ['Engine Type', 'RTO']: bangalore_cars_df[target_column] = bangalore_cars_df.groupby('bt')[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_columns = ['Mileage', 'Wheel Base']

for target_column in target_columns:
    # Convert the target column to numeric, coercing errors to NaN
    bangalore_cars_df[target_column] = pd.to_numeric(bangalore_cars_df[target_column], errors='coerce')
    
    # Perform group-wise mean imputation
    bangalore_cars_df[target_column] = bangalore_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))

group_column = 'bt'
target_columns = ['Steering Type', 'Front Brake Type', 'Rear Brake Type']
for target_column in ['Steering Type', 'Front Brake Type', 'Rear Brake Type']: bangalore_cars_df[target_column] = bangalore_cars_df.groupby('bt')[target_column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0]))

group_column = 'bt'
target_columns = ['Registration Year', 'Values per Cylinder', 'No of Cylinder',"Year of Manufacture"]

for target_column in target_columns:
    # Convert the target column to numeric, coercing errors to NaN
    bangalore_cars_df[target_column] = pd.to_numeric(bangalore_cars_df[target_column], errors='coerce')
    
    # Perform group-wise mean imputation and round up
    bangalore_cars_df[target_column] = bangalore_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()).apply(np.ceil))

group_column = 'bt'
target_columns = ['Height', 'Kerb Weight', 'Torque']

for target_column in target_columns:
    # Convert the target column to numeric, coercing errors to NaN
    bangalore_cars_df[target_column] = pd.to_numeric(bangalore_cars_df[target_column], errors='coerce')
    
    # Perform group-wise mean imputation
    bangalore_cars_df[target_column] = bangalore_cars_df.groupby(group_column)[target_column].transform(lambda x: x.fillna(x.mean()))
    
bangalore_cars_df.dropna(subset=['Fuel Suppy System', 'Value Configuration', 'Engine Displacement', 'Insurance Validity', 'Color', 'RTO', 'Super Charger', 'Mileage', 'Drive Type', 'Seating Capacity', 'Cargo Volumn'], inplace=True)

bangalore_cars_df.drop(columns=['new_car_overview', 'new_car_feature', 'new_car_specs', 'car_links'], inplace=True)


output_file_path = 'C:/Users/bhavan/Downloads/bangalore_cars_df.csv'
bangalore_cars_df.to_csv(output_file_path, index=False)


import streamlit as st

# Using the sidebar correctly
st.sidebar.title("Sidebar Example")
st.sidebar.button("Click Me")

