import numpy as np
import pandas as pd

def format_file(filename, extension):
    data = pd.read_fwf(filename)
    data.dropna()
    filename = filename.split(extension)[0]
    output = data.iloc[:, 2:6]
    output["cat"] = data['8']
    output.to_csv(filename + ".csv", index=False)

