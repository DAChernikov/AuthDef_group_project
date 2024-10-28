import csv
import json
import os

import boto3

BUCKET_NAME = 'auth-def-2024'
KEY_ID = os.environ.get("KEY_ID")
SECRET_ID = os.environ.get("SECRET_ID")


def s3_upload():
    session = boto3.session.Session()

    s3 = session.client(
        service_name='s3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=KEY_ID,
        aws_secret_access_key=SECRET_ID,
        region_name='ru-cental1',
    )

    dir_name = '../data/'
    file_names = os.listdir(dir_name)

    for file_name in file_names:
        if file_name.endswith('.txt'):
            meta_filename = file_name[:-3] + 'json'

            with open(dir_name + meta_filename, "r", encoding="utf-8") as json_file:
                json_data = json.load(json_file)

            original_name = json_data['link'].split('/')[-1].replace('.shtml', '.txt')

            s3.upload_file(dir_name + file_name, BUCKET_NAME, original_name)
            print(f"Загружен {original_name} ({file_name})")


def generate_meta_table():
    dir_name = '../data/'
    file_names = os.listdir(dir_name)
    file_names.sort()

    csv_data = [
        ['author', 'name', 'year', 'genres', 'annotation', 's3_link', 'local_link']
    ]

    for file_name in file_names:
        if file_name.endswith('.txt'):
            meta_filename = file_name[:-3] + 'json'

            with open(dir_name + meta_filename, "r", encoding="utf-8") as json_file:
                json_data = json.load(json_file)

            title = json_data['title'].replace("'", '').replace('"', '')

            original_name = json_data['link'].split('/')[-1].replace('.shtml', '.txt')

            annotation = json_data['annotation']
            if annotation:
                annotation = annotation.replace('\n', '')

            csv_data.append([
                json_data['author'],
                title,
                json_data['year'],
                json_data['genres'],
                annotation,
                f"https://storage.yandexcloud.net/{BUCKET_NAME}/{original_name}",
                f"{dir_name}{file_name}"
            ])

    with open("../meta_table.csv", mode="w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)


if __name__ == '__main__':
    s3_upload()
    generate_meta_table()
