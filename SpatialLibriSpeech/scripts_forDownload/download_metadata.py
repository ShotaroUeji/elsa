import urllib.request

url = "https://docs-assets.developer.apple.com/ml-research/datasets/spatial-librispeech/v1/metadata.parquet"
output_path = "metadata.parquet"

print(f"Downloading from {url}...")
urllib.request.urlretrieve(url, output_path)
print(f"Downloaded to {output_path}")
