import pandas as pd

# Load the dataset from the CSV file (replace 'your_data.csv' with the actual file path)
df = pd.read_csv('WORKLOAD_DF.csv')

# Percentage of data for training and testing
train_percentage = 0.8  # 80% for training
test_percentage = 0.2   # 20% for testing

# Calculate the number of rows for training data
train_size = int(len(df) * train_percentage)

# Split the data into training and test sets
train_data = df[:train_size]
test_data = df[train_size:]

# Reset the 'step' column in both datasets (starts from 1 again)
train_data['step'] = range(1, len(train_data) + 1)
test_data['step'] = range(1, len(test_data) + 1)

# Save the datasets to CSV files, removing the index column
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# Print confirmation
print("Training data saved to 'train_data.csv'")
print("Test data saved to 'test_data.csv'")
