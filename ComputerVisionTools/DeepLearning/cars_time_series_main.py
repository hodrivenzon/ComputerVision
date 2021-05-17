
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv(r"cars_count_files/cars_count_12_05_2021_22_36_13.csv")

    # df['date'].astype(np.datetime64)
    # now.strftime("%d_%m_%Y_%H_%M_%S")
    #
    #
    #
    # date_string = "21 June, 2018"
    #
    # print("date_string =", date_string)
    # print("type of date_string =", type(date_string))

    date_object = datetime.strptime(df['date'][0], "%d_%m_%Y_%H_%M_%S")

    df['date'] = [datetime.strptime(i, "%d_%m_%Y_%H_%M_%S") for i in df['date']]

    cars_per_min = df['cars_count'].groupby(df['date'].dt.to_period(freq='T')).sum()

    plt.plot(df['date'], df['cars_count'])
    plt.show()

    print()