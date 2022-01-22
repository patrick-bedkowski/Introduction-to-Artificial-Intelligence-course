import pandas as pd
from sklearn.datasets import load_wine

def load_data():
    data = load_wine(as_frame=True)
    return pd.concat([pd.DataFrame(data.data),data.target], axis=1)