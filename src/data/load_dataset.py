import wget
import os
import zipfile
import argparse


def load_dataset(
    url: str,
    output_path: str = './' 
):
    fname = wget.download(url, out=output_path)
    with zipfile.ZipFile(fname, 'r') as zip:
        zip.extractall(path=output_path)
    os.remove(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Zip archive dataset loader',
        description='Simple script for downloading zip dataset and unzipping it to specified folder'
    )

    parser.add_argument('-u', '--url', required=True, type=str, help='URL of zip file to download from')
    parser.add_argument('-p', '--path', type=str, default='./', help='Output path for the data')

    args = parser.parse_args()

    load_dataset(url=args.url, output_path=args.path)
