import numpy as np
import pandas as pd

def formatf(filename, extension = ".data"):
    cat = int(input("Insira a coluna categorica: "))
    num = input("Insira as colunas numericas: ").split(",")
    num = list(map(int, num))
    sep = input("Insira o separador: ")
    data = pd.read_csv(filename, sep=sep)
    data = data.replace("?", 0)
    data.dropna(inplace=True)
    filename = filename.split(extension)[0]
    output = data.iloc[:, num] #seleciona colunas de interesse
    size = 1000 if output.shape[0] > 1000 else output.shape[0]
    output = output.sample(size) #randomiza o dataset
    output=(output-output.mean())/output.std()
    output["cat"] = data.iloc[:,cat]
    output.to_csv(filename + ".csv", index=False)

