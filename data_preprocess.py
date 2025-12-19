import pandas as pd

df = pd.read_csv('data\WORKLOAD_DF.csv')

train_percentage = 0.8  # 80% for training
test_percentage = 0.2   # 20% for testing

train_size = int(len(df) * train_percentage)

train_data = df[:train_size]
test_data = df[train_size:]

train_data['step'] = range(1, len(train_data) + 1)
test_data['step'] = range(1, len(test_data) + 1)

train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("Training data saved to 'train_data.csv'")
print("Test data saved to 'test_data.csv'")
