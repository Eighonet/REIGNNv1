import os
from os import listdir
import subprocess
from tqdm import tqdm
import gzip
import shutil

def load() -> None:
    """
    Load and unpack Semantic Scholar Open Research Corpus. 
    
    """
    try:
        os.mkdir('tmp')
    except:
        pass
    
    try:
        os.mkdir('unpacked') 
    except:
        pass
    
    subprocess.run('aws s3 cp --no-sign-request --recursive s3://ai2-s2-research-public/open-corpus/2022-05-01/ ./tmp', shell=True, check=True)
    compressed = [f for f in listdir("./tmp") if f[-3:] == ".gz"]

    for file in tqdm(compressed):
        with gzip.open('./tmp/' + file, 'rb') as f_in:
            with open('./unpacked/' + file[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)