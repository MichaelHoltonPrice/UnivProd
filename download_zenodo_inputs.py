import os
import requests
import wget

# If necessary, create the directory for inputs downloaded from Zenodo
zenodo_inputs_dir = os.path.join(".", "zenodo_inputs")
if not os.path.isdir(zenodo_inputs_dir):
    os.mkdir(zenodo_inputs_dir)

# Get the base url (bucket_url) for downloading files form Zenodo for the
# UnivProd DOI 5510663
r = requests.get('https://zenodo.org/api/records/5510663')
bucket_url = r.json()["links"]["bucket"]

# Iterate over files that need downloading. Only download if the file has not
# already been downloaded.
files_to_download = ["CW2000.xlsx",
                     "CW2001.xlsx",
                     "CW2002.xlsx",
                     "CW2003.xlsx",
                     "CW2004.xlsx",
                     "CW2005.xlsx",
                     "CW2006.xlsx",
                     "CW2007.xlsx",
                     "CW2008.xlsx",
                     "CW2009.xlsx",
                     "CW2010.xlsx",
                     "CW2011.xlsx",
                     "CW2012.xlsx",
                     "CW2013.xlsx",
                     "CW2014.xlsx",
                     "CW2015.xlsx",
                     "CW2016.xlsx",
                     "CW2017.xlsx",
                     "CW2018.xlsx",
                     "delta_public_release_00_15.csv",
                     "delta_public_release_87_99.csv",
                     "mrc_table11.csv",
                     "mrc_table3.csv"]
for file_name in files_to_download:
    print("--")
    print(file_name)
    # The full path to the local file
    output_file = os.path.join(zenodo_inputs_dir, file_name)
    if not os.path.isfile(output_file):
        print("Downloading file")
        # The Zenodo URL for downloading the file
        file_url = bucket_url + "/" + file_name
        # Download the file using wget, and place it in zenodo_inputs_dir
        wget.download(file_url, out=zenodo_inputs_dir)
    else:
        print("File already downloaded")