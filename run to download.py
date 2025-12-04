import kagglehub

# Download latest version
path = kagglehub.dataset_download("keatonballard/synthetic-airline-passenger-and-flight-data")

print("Path to dataset files:", path)