import pandas as pd

# Load the original Excel file
input_file = r"C:\Users\mvalois\Desktop\CurrentProjects\AGISTIN\GitHub Repositories\Uni-Kassel_Task-3.2\agistin\main\Cases\Market_Model\RESULT_OVERVIEW_CAPACITY_MARKET_FCR_2023-01-01_2023-12-31.xlsx"  # Replace with your actual file path
output_file = r"C:\Users\mvalois\Desktop\CurrentProjects\AGISTIN\GitHub Repositories\Uni-Kassel_Task-3.2\agistin\main\Cases\Market_Model\RESULT_OVERVIEW_CAPACITY_MARKET_FCR_2023-01-01_2023-12-31_15Minutes.xlsx"  # The name for the output file

# Load the data
df = pd.read_excel(input_file)

# Create a new DataFrame to store the expanded data
expanded_data = []

# Loop through each row in the original data
for index, row in df.iterrows():
    # Get the price for the 4-hour block
    four_hour_price = row['GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]']

    # Calculate the 15-minute price by dividing by 16
    fifteen_min_price = four_hour_price / 16

    # Expand this price into 16 rows, each representing a 15-minute block
    for i in range(16):
        expanded_data.append({
            'Original_Index': index,  # To keep track of the original row
            '15_Min_Block': i + 1,
            'Price_[EUR/MW]': fifteen_min_price
        })

# Convert the expanded data into a DataFrame
expanded_df = pd.DataFrame(expanded_data)

# Save the expanded data to a new Excel file
expanded_df.to_excel(output_file, index=False)

print("15-minute block prices calculated and saved to", output_file)