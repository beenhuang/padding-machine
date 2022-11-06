#!/usr/bin/env python3

"""
<file>    plot.py
<brief>   
"""

import matplotlib.pyplot as plt


def show_bar_plot(x, y, plt_title, x_label, y_label):
    #fig, ax = plt.subplots()
    #ax.get_yaxis().get_major_formatter().set_scientific(False)

    #p = plt.bar(x, y, width = 0.5, tick_label=x)
    p = plt.bar(x, y, tick_label=x)
    plt.bar_label(p, label_type='edge')

    plt.title(plt_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()     

#
def save_bar_plot(x, y, file, plt_title, x_label, y_label):
    p = plt.bar(x, y, tick_label=x)
    plt.bar_label(p, label_type='edge')

    plt.title(plt_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(file)
    plt.close()





if __name__ == "__main__":
    pass

