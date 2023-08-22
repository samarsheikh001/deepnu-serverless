import os
import boto3

from dotenv import load_dotenv
load_dotenv()


s3 = boto3.resource(
    's3',
    endpoint_url=os.getenv("S3_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY")
)

bucket = s3.Bucket('models')


def upload_file_to_s3(file_path, s3_key):
    try:
        bucket.upload_file(file_path, s3_key)
        print("File uploaded successfully!")
    except Exception as e:
        print(f"Error uploading file: {e}")
        raise


def download_file_from_s3(s3_key, file_path):
    try:
        bucket.download_file(s3_key, file_path)
        print("File downloaded successfully!")
    except Exception as e:
        print(f"Error downloading file: {e}")


# upload_file_to_s3("model/uber_inpainting.safetensors",
#                   "uber_inpainting.safetensors")

# download_file_from_s3(
#     "cloth_segm.pth", "cloth_segm.pth")
