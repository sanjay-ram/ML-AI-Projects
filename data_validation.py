import pandas as pd

# Create and save a sample dataset
data = {
    "customer_id": [1, 2, 3, 4, 5],
    "membership_level": ["Bronze", "Silver", "Gold", "Silver", "Bronze"]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_filename = "customer_data.csv"
df.to_csv(csv_filename, index=False)

# Load the dataset
incoming_data = pd.read_csv(csv_filename)

# Validate the data
def validate_data(df):
    # Check for null values in 'customer_id'
    if df['customer_id'].isnull().any():
        print("Validation Error: 'customer_id' column contains null values.")
    else:
        print("No null values in 'customer_id' column.")

    # Check that 'membership_level' contains only allowed values
    allowed_values = {"Bronze", "Silver", "Gold"}
    invalid_values = set(df['membership_level']) - allowed_values
    if invalid_values:
        print(f"Validation Error: 'membership_level' contains invalid values: {invalid_values}")
    else:
        print("All values in 'membership_level' are valid.")

# Run validation
validate_data(incoming_data)