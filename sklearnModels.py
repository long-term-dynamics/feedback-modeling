import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas
import sys

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

# Usage:
# python3 sklearnModels.py voting

# polynomial interpolation constant
DEGREE = 3

def main():

    if len(sys.argv) < 2:
        # run all experiments by default
        do_experiment('voting')
        do_experiment('income')
        do_experiment('pay')
    else:
        data = sys.argv[1]
        do_experiment(data)

def do_experiment(data):
    if data == 'voting':
        filename = 'data/BlackWhiteVotingPercentageUS.csv'
        year_col = 'Year Voted'
        num_col = 'White Total Population'
        denom_col = 'Black Total Population'
        outfile_stub = 'sklearn_votinggap_ratio'
        flip = True
    elif data == 'income':
        filename = 'data/PicketySaezIncomeTopDecile.csv'
        year_col = 'Year'
        num_col = 'Percent Income in Top Decile'
        denom_col = '10'
        outfile_stub = 'sklearn_incomegap_ratio'
        flip = False
    elif data == 'pay':
        filename = 'data/GenderPayGapData.csv'
        year_col = 'Year From Start'
        num_col = 'x'
        denom_col = 'do not use'
        outfile_stub = 'sklearn_paygap_ratio'
        flip = True

    print('\n------------------------\n' + data + ": HALF training")
    outfile = outfile_stub + '_half.png'
    train_and_plot(filename, year_col, num_col, denom_col, 0.5, data, flip, outfile)
    print('\n' + data + ": 2/3 training")
    outfile = outfile_stub + '_twothirds.png'
    train_and_plot(filename, year_col, num_col, denom_col, 0.66, data, flip, outfile)

def train_and_plot(filename, year_col, num_col, denom_col, train_perct, data, flip, outfile):
    all_years, real_y, x_train, y_train, x_test, y_test = \
        get_training_test(filename, year_col, num_col, denom_col, train_perct, data, flip)

    reg = LinearRegression().fit(x_train, y_train)
    linreg_y_hat = reg.predict(x_test)
    rmse = mean_squared_error(y_test, linreg_y_hat, squared=False)
    print("Linear Regression RMSE:" + str(rmse))

    poly_model = make_pipeline(PolynomialFeatures(DEGREE), Ridge())
    poly_model.fit(x_train, y_train)
    poly_y_hat = poly_model.predict(x_test)
    rmse = mean_squared_error(y_test, poly_y_hat, squared=False)
    print("Poly Interp RMSE:" + str(rmse))

    dt_model = DecisionTreeRegressor(random_state=0).fit(x_train, y_train)
    dt_y_hat = dt_model.predict(x_test)
    rmse = mean_squared_error(y_test, dt_y_hat, squared=False)
    print("Decision Tree RMSE:" + str(rmse))

    if data == 'voting':
        ytitle = 'White / Black Voting %'
        xticks = 8
    elif data == 'income':
        ytitle = 'Total Income of Top 10% / 10%'
        xticks = 10
    else: # data = pay
        ytitle = "Men's / Women's Earnings"
        xticks = 8

    plot_and_save(all_years, real_y, linreg_y_hat, poly_y_hat, dt_y_hat, xticks, ytitle, outfile)

def get_training_test(filename, x_col, col_numerator, col_demonimator, train_percent, data = None, flip = True):
    df = pandas.read_csv(filename)
    x = df[x_col].to_numpy()
    x = np.reshape(x, (-1, 1))
    if data == 'pay':
        # the values in the csv are (1 - women's / men's earnings).
        # we want (men's earnings / women's earnings) - 1.0
        x = x + 1960
        y = 1.0/(1.0 - df[col_numerator])
    else:
        y = df[col_numerator] / df[col_demonimator]
    if flip:
        x = np.flip(x)
        y = np.flip(y)

    if data == 'income':
        # start at year 1945
        x = x[28:]
        y = y[28:]
    train_stop = math.floor(len(x) * train_percent)
    x_train = x[: train_stop]
    x_test = x[train_stop : ]
    y_train = y[ : train_stop]
    y_test = y[train_stop : ]
    return x, y, x_train, y_train, x_test, y_test

def plot_and_save(all_years, real_y, linreg, poly, dt, xticks, ytitle, outfile):
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)

    len_reg = len(linreg)
    len_poly = len(poly)
    len_dt = len(dt)

    plt.ion()
    plt.figure(1)

    plt.plot(all_years, real_y, 'g-o', linewidth=2, label='Real data')
    plt.plot([min(all_years)-0.5, max(all_years)+0.5], [1, 1], 'k-', linewidth=1)
    plt.plot(all_years[-len_reg:], linreg, 'red', linewidth=1, label='Linear regression')
    plt.plot(all_years[-len_poly:], poly, 'purple', linewidth=1, label='Polynomial interpolation')
    plt.plot(all_years[-len_dt:], dt, 'darkblue', linewidth=1, label='Decision tree regressor')
    plt.xlim(min(all_years)-0.5, max(all_years)+0.5)
    plt.xlabel("Year", fontsize=20)
    plt.ylabel(ytitle, fontsize=20)
    plt.xticks(range(int(min(all_years)), int(max(all_years)), xticks))
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.show()
    plt.savefig(outfile)
    plt.clf()
    #plt.legend(loc='lower left', fontsize=14)
    #plt.ylim(0.8, 1.5)
    #plt.show()


if __name__ == "__main__":
    main()
