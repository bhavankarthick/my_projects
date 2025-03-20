import pandas as pd

# Load the customer data with a different encoding
customers_df = pd.read_csv('C:/Users/bhavan/Downloads/customers.csv', encoding='ISO-8859-1')

# Load the product data
products_df = pd.read_csv('C:/Users/bhavan/Downloads/products.csv', encoding='ISO-8859-1')

# Load the sales data
sales_df = pd.read_csv('C:/Users/bhavan/Downloads/sales.csv', encoding='ISO-8859-1')

# Load the store data
stores_df = pd.read_csv('C:/Users/bhavan/Downloads/stores.csv', encoding='ISO-8859-1')

# Load the currency exchange rates data
exchange_rates_df = pd.read_csv('C:/Users/bhavan/Downloads/exchange_rates.csv', encoding='ISO-8859-1')

print(stores_df.head(5))

# Check for missing values in each DataFrame

# Missing values in the Customers DataFrame
print("Missing values in Customers DataFrame:")
print(customers_df.isnull().sum())

# Remove rows with missing values in the Customers DataFrame
customers_df_cleaned = customers_df.dropna()

# Optional: Check the number of rows before and after cleaning
print(f"Original number of rows: {len(customers_df)}")
print(f"Number of rows after removing missing values: {len(customers_df_cleaned)}")

# Convert the 'Birthday' column to datetime
customers_df_cleaned['Birthday'] = pd.to_datetime(customers_df_cleaned['Birthday'], errors='coerce')

# Check the updated data types
print("\nUpdated Data Types in Customers DataFrame:")
print(customers_df_cleaned.dtypes)

# Missing values in the Products DataFrame
print("\nMissing values in Products DataFrame:")
print(products_df.isnull().sum())

print(products_df.dtypes)

products_df['Unit Cost USD'] = pd.to_numeric(products_df['Unit Cost USD'].str.replace('$', '').str.replace(',', ''), errors='coerce')
products_df['Unit Price USD'] = pd.to_numeric(products_df['Unit Price USD'].str.replace('$', '').str.replace(',', ''), errors='coerce')
print(products_df.dtypes)

print(sales_df.isnull().sum())
# Remove the 'Delivery Date' column from the sales_df DataFrame
sales_df = sales_df.drop(columns=['Delivery Date'])

# Verify that the column has been removed by printing the column names
print("\nColumns in Sales DataFrame after removing 'Delivery Date':")
print(sales_df.columns)
# Check the column data types in the Sales DataFrame
print("Data Types in Sales DataFrame:")
print(sales_df.dtypes)

# Convert 'Order Date' from object to datetime
# Ensure you are handling any potential invalid date formats
sales_df['Order Date'] = pd.to_datetime(sales_df['Order Date'], errors='coerce')

# Optionally, convert 'Currency Code' to category (if needed)
sales_df['Currency Code'] = sales_df['Currency Code'].astype('category')

# Check the data types after the conversion
print("\nData Types After Conversion:")
print(sales_df.dtypes)

print(stores_df.isnull().sum())
stores_df_cleaned = stores_df.dropna()
print(stores_df_cleaned.isnull().sum())

# Try converting 'Open Date' without specifying format
stores_df['Open Date'] = pd.to_datetime(stores_df['Open Date'], errors='coerce')
print(stores_df.dtypes)

print(exchange_rates_df.isnull().sum())
print(exchange_rates_df.dtypes)
exchange_rates_df['Date'] = pd.to_datetime(exchange_rates_df['Date'], errors='coerce')
print(exchange_rates_df.dtypes)




sales_with_customers = pd.merge(sales_df, customers_df, on='CustomerKey')
sales_with_products = pd.merge(sales_with_customers, products_df, on='ProductKey')
print(sales_with_customers.head())
print(sales_with_customers.columns)
null_values = sales_with_customers.isnull().sum()
print(sales_with_customers.columns)

from sqlalchemy import create_engine

# Define your MySQL database connection parameters
username = 'root'  # Your MySQL username
password = 'bhasaraj'  # Your MySQL password
host = 'localhost'  # or your MySQL server IP
database = 'sales_with_customers'   # Your database name

# Create a connection to the MySQL database
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}/{database}')

# Write the merged DataFrame to MySQL
sales_with_customers.to_sql('sales_with_customers', con=engine, index=False, if_exists='replace')

print("Data transferred successfully!")

from sqlalchemy import create_engine
import pymysql  # Ensure pymysql is imported

# Define your MySQL database connection parameters
username = 'root'  # Your MySQL username
password = 'bhasaraj'  # Your MySQL password
host = 'localhost'  # or your MySQL server IP
database = 'sales_with_products'  # Your actual database name

# Create a connection to the MySQL database
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}/{database}')

# Write the merged DataFrame to MySQL
sales_with_products.to_sql('sales_with_products', con=engine, index=False, if_exists='replace')

# Assuming sales_with_products is your DataFrame
download_path = r'C:\Users\bhavan\Downloads\sales_with_products.csv'  # For Windows
# Or for Mac/Linux: download_path = '/Users/bhavan/Downloads/sales_with_products.csv'

# Save DataFrame to CSV
sales_with_products.to_csv(download_path, index=False)

print(f"File saved to: {download_path}")

merged_df = pd.merge(sales_df, customers_df, on='CustomerKey', how='left')

# Merge Sales with Products on ProductKey
sales_with_products_df = pd.merge(sales_df, products_df, on='ProductKey', how='inner')

# Check the first few rows to ensure the merge was successful
print(sales_with_products_df.head())

# Check the column names of the merged DataFrame
print(sales_with_products_df.columns)

# Save the merged DataFrame to a CSV file in your local Downloads folder
sales_with_products_df.to_csv('C:/Users/bhavan/Downloads/sales_with_products.csv', index=False)

# Merge the sales and store DataFrames
sales_with_stores = pd.merge(sales_df, stores_df, on='StoreKey', how='inner')

# Display the column names of the merged DataFrame
print(sales_with_stores.columns)

from sqlalchemy import create_engine
import pymysql
import pandas as pd

# Define your MySQL database connection parameters
username = 'root'  # Your MySQL username
password = 'bhasaraj'  # Your MySQL password
host = 'localhost'  # Your MySQL server address
database = 'sales_with_stores'  # Your actual database name

# Create a connection to the MySQL database
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}/{database}')

# Transfer the sales_with_store DataFrame to MySQL
# Assuming sales_with_store is your merged DataFrame
sales_with_stores.to_sql('sales_with_store', con=engine, index=False, if_exists='replace')

print("DataFrame has been transferred to MySQL database!")

# Save the DataFrame as a CSV file on your computer
sales_with_stores.to_csv('C:/Users/bhavan/Downloads/sales_with_stores.csv', index=False)

print("DataFrame has been saved as 'sales_with_stores.csv' in your Downloads folder.")

sales_with_stores = sales_with_stores.drop(columns=['Square Meters'])

# Merge sales and product data on a common key, assuming 'ProductKey' is the common column
sales_with_products_df = pd.merge(sales_df, products_df, on='ProductKey', how='left')

# Display the merged DataFrame
print(sales_with_products_df.columns)

# Assuming sales_df and exchange_rates_df are already loaded

# Get the first 200 rows of the sales dataframe
sales_subset = sales_df.head(200)

# Merge the first 200 rows of sales with the exchange rates on the specified columns
merged_df = pd.merge(
    sales_subset, 
    exchange_rates_df, 
    left_on='Currency Code',  # Column from sales_df
    right_on='Currency',      # Column from exchange_rates_df
    how='left'
)

# Display the merged dataframe
print(merged_df.head())

# Specify the file path where you want to save the merged dataframe
output_file_path = 'C:/Users/bhavan/Downloads/merged_sales_exchange_rates.csv'

# Save the merged dataframe to a CSV file
merged_df.to_csv(output_file_path, index=False)

print(f'Merged dataframe saved to {output_file_path}')

import pandas as pd
from sqlalchemy import create_engine

# Assuming merged_df is already created

# Database connection details
username = 'root'   # Replace with your MySQL username
password = 'bhasaraj'     # Replace with your MySQL password
host = 'localhost'            # Replace with your MySQL host
database = 'merged_df '            # Replace with your database name

# Create SQLAlchemy engine
engine = create_engine(f'mysql+mysqlconnector://{username}:{password}@{host}/{database}')

# Export merged_df to a MySQL table called 'merged_sales_exchange_rates'
merged_df.to_sql('merged_sales_exchange_rates', con=engine, if_exists='replace', index=False)

print('DataFrame exported to MySQL successfully.')

# Define the file path
file_path = 'C:/Users/bhavan/Downloads/Stores.csv'

# Load the CSV into a pandas DataFrame
Stores_stores_df = pd.read_csv(file_path)

# Check the first few rows to verify
print(stores_df.head())

store_analysis = pd.merge(sales_df, Stores_stores_df, on='StoreKey', how='inner')

# Check the first few rows of the merged DataFrame
print(store_analysis.head())

# Save the merged DataFrame to a CSV file
store_analysis.to_csv('C:/Users/bhavan/Downloads/store_analysis.csv', index=False)

import pandas as pd
from sqlalchemy import create_engine

# Assuming your MySQL credentials are as follows
username = 'root'  # Replace with your MySQL username
password = 'bhasaraj'  # Replace with your MySQL password
host = 'localhost'          # Replace with your MySQL host (or 'localhost' if running locally)
database = 'store_analysis'  # Replace with your target database name

# Create a connection string for MySQL
connection_string = f'mysql+pymysql://{username}:{password}@{host}/{database}'

# Create an engine to connect to the MySQL database
engine = create_engine(connection_string)

# Export the DataFrame to MySQL
store_analysis.to_sql('store_analysis', con=engine, if_exists='replace', index=False)

print("Data exported successfully!")



