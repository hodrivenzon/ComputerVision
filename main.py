

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    import pandas as pd

    # url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/US_Crime_Rates/US_Crime_Rates_1960_2014.csv"
    # crime = pd.read_csv(url, sep=',')

    # url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'
    # users = pd.read_csv(url, sep='|')
    df = pd.read_csv(r"C:\Users\hodda\Downloads\NationalNames.csv\NationalNames.csv")

    # y = users.groupby('occupation').age.mean()
    print()
