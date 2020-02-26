#!/usr/bin/env python3
from pprz_data.pprz_data import DATA

def plot_all(data):
    import matplotlib
    import matplotlib.pyplot as plt
    # %config InlineBackend.figure_format = 'retina'
    import matplotlib as mpl
    mpl.style.use('default')
    import seaborn #plotting lib, but just adding makes the matplotlob plots better

    # fig=plt.figure(figsize=(19,7))
    # df_labelled.plot(y=['m1', 'alt'], figsize=(17,7));plt.show()
    data.plot(subplots=True, figsize=(12,10));plt.show()
    
def main():
    ac_id = '9'
    filename = '../data/jumper_2nd.data'
    data = DATA(filename, ac_id, data_type='fault')
    plot_all(data.df_All)


if __name__ == "__main__":
    main()