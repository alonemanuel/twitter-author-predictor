import pandas as pd
from task1.src.classifier import classify


if __name__ == '__main__':
    data = pd.read_csv('test.csv')
    print(classify(data))
