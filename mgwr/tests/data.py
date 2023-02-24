"""
this script formats data before executing mgwr.
"""

import numpy as NUM
import pandas as pd
import os

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def get_test2021_sub_xxx_10(xxx, nx=11):
    depVarName = 'Y_new'
    indVarNames = 'X0, X1, X2, X3, X4, X5, X6, X7, X8, X9'.split(", ")

    if nx-1 > len(indVarNames):
        nx = len(indVarNames) + 1
    if nx < 2:
        nx = 2

    data_path = os.path.join(data_dir, "test2021_sub_"+str(xxx)+"_10.csv")
    df = pd.read_csv(data_path)
    indVarNames = indVarNames[:(nx-1)]

    n = df.shape[0]
    k = len(indVarNames)

    y = df[depVarName].values.reshape((-1,1))
    x = NUM.ones((n, k+1), dtype=float)

    for column, variable in enumerate(indVarNames):
        x[:, column + 1] = df[[variable]].values.flatten()

    ### follow the projection in the paper
    coords = list(zip(df['x_coord_earth'], df['y_coord_earth']))
    coords = NUM.asarray(coords)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    x[:, 1:] = (x[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)
    return (x, y, coords, data_path)

def get_test0314_sub_xxx_10(xxx, nx=11):
    depVarName = 'Y_new'
    indVarNames = 'X0, X1, X2, X3, X4, X5, X6, X7, X8, X9'.split(", ")

    if nx-1 > len(indVarNames):
        nx = len(indVarNames) + 1
    if nx < 2:
        nx = 2

    data_path = os.path.join(data_dir, "test0314_sub_"+str(xxx)+"_10.csv")
    df = pd.read_csv(data_path)
    indVarNames = indVarNames[:(nx-1)]

    n = df.shape[0]
    k = len(indVarNames)

    y = df[depVarName].values.reshape((-1,1))
    x = NUM.ones((n, k+1), dtype=float)

    for column, variable in enumerate(indVarNames):
        x[:, column + 1] = df[[variable]].values.flatten()

    ### follow the projection in the paper
    coords = list(zip(df['x_coord'], df['y_coord']))
    coords = NUM.asarray(coords)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    x[:, 1:] = (x[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)
    return (x, y, coords, data_path)


def get_kc_house_price(nx=2):
    depVarName = 'price_Z_SCORE'
    indVarNames = ['sqft_living_Z_SCORE']

    if nx-1 > len(indVarNames):
        nx = len(indVarNames) + 1
    if nx < 2:
        nx = 2

    data_path = os.path.join(data_dir, "kc_house_data_2.csv")
    df = pd.read_csv(data_path)
    df = df.iloc[:7000,:]
    indVarNames = indVarNames[:(nx-1)]

    n = df.shape[0]
    k = len(indVarNames)

    y = df[depVarName].values.reshape((-1,1))
    x = NUM.ones((n, k+1), dtype=float)

    for column, variable in enumerate(indVarNames):
        x[:, column + 1] = df[[variable]].values.flatten()

    coords = list(zip(df['XCoord'], df['YCoord']))
    coords = NUM.asarray(coords)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    x[:, 1:] = (x[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)
    return (x, y, coords, data_path)

def get_simulation_design1_mini(nx=3):
    depVarName = 'y'
    indVarNames = ['x1', 'x2']

    if nx-1 > len(indVarNames):
        nx = len(indVarNames) + 1
    if nx < 2:
        nx = 2

    data_path = os.path.join(data_dir, "simulation_design1_mini.csv")
    df = pd.read_csv(data_path)

    indVarNames = indVarNames[:(nx-1)]

    n = df.shape[0]
    k = len(indVarNames)

    y = df[depVarName].values.reshape((-1,1))
    x = NUM.ones((n, k+1), dtype=float)

    for column, variable in enumerate(indVarNames):
        x[:, column + 1] = df[[variable]].values.flatten()

    coords = list(zip(df['xcoord'], df['ycoord']))
    coords = NUM.asarray(coords)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    x[:, 1:] = (x[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)
    return (x, y, coords, data_path)

def get_simulation_design1(nx=3):
    depVarName = 'y'
    indVarNames = ['x1', 'x2']

    if nx-1 > len(indVarNames):
        nx = len(indVarNames) + 1
    if nx < 2:
        nx = 2

    data_path = os.path.join(data_dir, "simulation_design1.csv")
    df = pd.read_csv(data_path)

    indVarNames = indVarNames[:(nx-1)]

    n = df.shape[0]
    k = len(indVarNames)

    y = df[depVarName].values.reshape((-1,1))
    x = NUM.ones((n, k+1), dtype=float)

    for column, variable in enumerate(indVarNames):
        x[:, column + 1] = df[[variable]].values.flatten()

    coords = list(zip(df['xcoord'], df['ycoord']))
    coords = NUM.asarray(coords)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    x[:, 1:] = (x[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)
    return (x, y, coords, data_path)

def get_georgia_data(nx=2):
    """
    :param nx, number of x variables, including intercept, ranging 2-6
    :return: a test data, both x and y are centered, has intercept column
    """
    depVarName = 'PctBach'
    indVarNames = ['PctFB', 'PctBlack', 'PctRural', 'PctEld', 'PctPov']

    if nx-1 > len(indVarNames):
        nx = len(indVarNames) + 1
    if nx < 2:
        nx = 2

    data_path = os.path.join(data_dir, "GData_utm.csv")
    georgia_data = pd.read_csv(data_path)

    indVarNames = indVarNames[:(nx-1)]

    n = georgia_data.shape[0]
    k = len(indVarNames)
    assert k == nx - 1

    y = georgia_data[depVarName].values.reshape((-1,1))
    x = NUM.ones((n, k+1), dtype=float)

    for column, variable in enumerate(indVarNames):
        x[:, column + 1] = georgia_data[[variable]].values.flatten()

    coords = list(zip(georgia_data['X'], georgia_data['Y']))
    coords = NUM.asarray(coords)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    x[:, 1:] = (x[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)

    return (x, y, coords, data_path)

def get_meld_6830_data(nx=7):
    depVarName = 'Price'
    indVarNames = ['Rooms',  'Bathroom', 'Car', 'Landsize', 'BuildingArea'] # 'Distance',

    if nx-1 > len(indVarNames):
        nx = len(indVarNames) + 1
    if nx < 2:
        nx = 2

    data_path = os.path.join(data_dir, "melb_data_6830_centered.csv")
    dt = pd.read_csv(data_path)

    indVarNames = indVarNames[:(nx-1)]

    n = dt.shape[0]
    k = len(indVarNames)
    assert k == nx - 1

    y = dt[depVarName].values.reshape((-1,1))
    x = NUM.ones((n, k+1), dtype=float)

    for column, variable in enumerate(indVarNames):
        x[:, column + 1] = dt[[variable]].values.flatten()

    coords = list(zip(dt['Longtitude'], dt['Lattitude']))
    coords = NUM.asarray(coords)

    return (x, y, coords, data_path)

def get_meld_1122_data(nx=7):
    depVarName = 'Price'
    indVarNames = ['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize', 'BuildingArea']

    if nx-1 > len(indVarNames):
        nx = len(indVarNames) + 1
    if nx < 2:
        nx = 2

    data_path = os.path.join(data_dir, "melb_data_1122_centered.csv")
    dt = pd.read_csv(data_path)

    indVarNames = indVarNames[:(nx-1)]
    # indVarNames = indVarNames[:nx]

    n = dt.shape[0]
    k = len(indVarNames)
    assert k == nx - 1
    # assert k == nx

    y = dt[depVarName].values.reshape((-1,1))
    x = NUM.ones((n, k+1), dtype=float)
    # x = NUM.ones((n, k), dtype=float)

    for column, variable in enumerate(indVarNames):
        x[:, column + 1] = dt[[variable]].values.flatten()
        # x[:, column] = dt[[variable]].values.flatten()

    coords = list(zip(dt['Longtitude'], dt['Lattitude']))
    coords = NUM.asarray(coords)

    return (x, y, coords, data_path)

def get_meld_279_data(nx=7):
    depVarName = 'Price'.upper()
    indVarNames = 'Rooms;Distance;Bathroom;Car;Landsize;BuildingArea'.upper().split(";")

    if nx-1 > len(indVarNames):
        nx = len(indVarNames) + 1
    if nx < 2:
        nx = 2

    data_path = os.path.join(data_dir, "melb_data_279.csv")
    dt = pd.read_csv(data_path)

    indVarNames = indVarNames[:(nx-1)]
    # indVarNames = indVarNames[:nx]

    n = dt.shape[0]
    k = len(indVarNames)
    assert k == nx - 1
    # assert k == nx

    y = dt[depVarName].values.reshape((-1,1))
    x = NUM.ones((n, k+1), dtype=float)
    # x = NUM.ones((n, k), dtype=float)

    for column, variable in enumerate(indVarNames):
        x[:, column + 1] = dt[[variable]].values.flatten()
        # x[:, column] = dt[[variable]].values.flatten()

    coords = list(zip(dt['Longtitude'], dt['Lattitude']))
    coords = NUM.asarray(coords)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    x[:, 1:] = (x[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)

    return (x, y, coords, data_path)

def get_finalprojected_data(nx=2, sample=200):
    data_path = os.path.join(data_dir, "finalprojected_select.csv")
    df = pd.read_csv(data_path)
    df = df.iloc[:sample,:]
    depVarName = 'GROWTH_BI'
    indVarNames = ['LOGPCR69', 'PCR1969', 'PCR1970', 'PCR1971',
                   'PCR1972', 'PCR1973', 'PCR1974', 'PCR1975', 'PCR1976', 'PCR1977',
                   'PCR1978', 'PCR1979', 'PCR1980', 'PCR1981', 'PCR1982', 'PCR1983',
                   'PCR1984', 'PCR1985', 'PCR1986', 'PCR1987', 'PCR1988', 'PCR1989',
                   'PCR1990', 'PCR1991', 'PCR1992', 'PCR1993', 'PCR1994', 'PCR1995',
                   'PCR1996', 'PCR1997', 'PCR1998', 'PCR1999', 'PCR2000', 'PCR2001',
                   'PCR2002', 'POP1969', 'POP1970', 'POP1971', 'POP1972', 'POP1973',
                   'POP1974', 'POP1975', 'POP1976', 'POP1977', 'POP1978', 'POP1979',
                   'POP1980', 'POP1981', 'POP1982', 'POP1983', 'POP1984', 'POP1985',
                   'POP1986', 'POP1987', 'POP1988', 'POP1989', 'POP1990', 'POP1991',
                   'POP1992', 'POP1993', 'POP1994', 'POP1995', 'POP1996', 'POP1997',
                   'POP1998', 'POP1999', 'POP2000', 'POP2001', 'POP2002', 'GROWTH',
                   'LOGPCR69']

    if nx-1 > len(indVarNames):
        nx = len(indVarNames) + 1
    if nx < 2:
        nx = 2

    indVarNames = indVarNames[:(nx - 1)]

    n = df.shape[0]
    k = len(indVarNames)

    y = df[depVarName].values.reshape((-1,1)).astype(float)
    x = NUM.ones((n, k+1), dtype=float)

    for column, variable in enumerate(indVarNames):
        x[:, column + 1] = df[[variable]].values.flatten()

    coords = list(zip(df['XCoord'], df['YCoord']))
    coords = NUM.asarray(coords)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    x[:, 1:] = (x[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)

    return (x, y, coords, data_path)

def get_finalprojected_data_all(nx=2):
    data_path = os.path.join(data_dir, "finalprojected_select.csv")
    df = pd.read_csv(data_path)
    depVarName = 'GROWTH_BI'
    indVarNames = ['LOGPCR69', 'PCR1969', 'PCR1970', 'PCR1971',
                   'PCR1972', 'PCR1973', 'PCR1974', 'PCR1975', 'PCR1976', 'PCR1977',
                   'PCR1978', 'PCR1979', 'PCR1980', 'PCR1981', 'PCR1982', 'PCR1983',
                   'PCR1984', 'PCR1985', 'PCR1986', 'PCR1987', 'PCR1988', 'PCR1989',
                   'PCR1990', 'PCR1991', 'PCR1992', 'PCR1993', 'PCR1994', 'PCR1995',
                   'PCR1996', 'PCR1997', 'PCR1998', 'PCR1999', 'PCR2000', 'PCR2001',
                   'PCR2002', 'POP1969', 'POP1970', 'POP1971', 'POP1972', 'POP1973',
                   'POP1974', 'POP1975', 'POP1976', 'POP1977', 'POP1978', 'POP1979',
                   'POP1980', 'POP1981', 'POP1982', 'POP1983', 'POP1984', 'POP1985',
                   'POP1986', 'POP1987', 'POP1988', 'POP1989', 'POP1990', 'POP1991',
                   'POP1992', 'POP1993', 'POP1994', 'POP1995', 'POP1996', 'POP1997',
                   'POP1998', 'POP1999', 'POP2000', 'POP2001', 'POP2002', 'GROWTH',
                   'LOGPCR69']

    if nx-1 > len(indVarNames):
        nx = len(indVarNames) + 1
    if nx < 2:
        nx = 2

    indVarNames = indVarNames[:(nx - 1)]

    n = df.shape[0]
    k = len(indVarNames)

    y = df[depVarName].values.reshape((-1,1)).astype(float)
    x = NUM.ones((n, k+1), dtype=float)

    for column, variable in enumerate(indVarNames):
        x[:, column + 1] = df[[variable]].values.flatten()

    coords = list(zip(df['XCoord'], df['YCoord']))
    coords = NUM.asarray(coords)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    x[:, 1:] = (x[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)

    return (x, y, coords, data_path)

def get_random_279_data(nx=2):
    data_path = os.path.join(data_dir, "random_279.csv")
    df = pd.read_csv(data_path)

    depVarName = 'dependent'
    indVarNames = ['explanatory1', 'explanatory2', 'explanatory3' , 'explanatory4', 'explanatory5']

    if nx - 1 > len(indVarNames):
        nx = len(indVarNames) + 1
    if nx < 2:
        nx = 2

    indVarNames = indVarNames[:(nx - 1)]

    n = df.shape[0]
    k = len(indVarNames)

    y = df[depVarName].values.reshape((-1,1)).astype(float)
    x = NUM.ones((n, k+1), dtype=float)
    for column, variable in enumerate(indVarNames):
        varData = df[[variable]].values
        x[:, column + 1] = varData.flatten()

    coords = list(zip(df['x'], df['y']))
    coords = NUM.asarray(coords)
    # y = (y - y.mean(axis=0)) / y.std(axis=0)
    # x[:, 1:] = (x[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)

    return (x, y, coords, data_path)

############################## neighbors and kernel weights ####################################
### this function find neighbors and calculate kernel weights for testing purpose, we don't have to re-program this
def distanceMatrix(coords, bw, index=None, gradient=False, weight_scheme="bisquare"):
    coords = NUM.array(coords)
    xx = coords[:,0]
    yy = coords[:,1]
    n = len(xx)
    distance_matrix = NUM.sqrt(NUM.add(xx.reshape((-1, 1)), -xx.reshape((1, -1))) ** 2 + \
                      NUM.add(yy.reshape((-1, 1)), -yy.reshape((1, -1))) ** 2)
    sort_matrix = NUM.argsort(distance_matrix, axis=1)
    # bw_dist = distance_matrix[NUM.arange(n), sort_matrix[:, (bw-1)]].reshape((-1,1)) # nNeibhors does not include the point
    bw_dist = distance_matrix[NUM.arange(n), sort_matrix[:, bw]].reshape((-1, 1))  # nNeibhors includes the point
    if weight_scheme=="bisquare":
        weight_matrix = pow(1.0 - pow((distance_matrix/bw_dist), 2.0), 2.0)
    elif weight_scheme=="gaussian":
        weight_matrix = NUM.exp(- pow(distance_matrix / bw_dist, 2.0) / 2.)

    weight_matrix[NUM.arange(n).reshape((-1, 1)), sort_matrix[:, bw:]] = 0
    # assert NUM.all(NUM.sum(distance_matrix / bw_dist < 1, axis=1) == bw)
    # assert NUM.all(NUM.sum(weight_matrix > 0, axis=1) == bw)

    if gradient:
        gradient_matrix = 4 * (1.0 - pow(distance_matrix/bw_dist, 2.0)) * pow(distance_matrix, 2) / pow(bw_dist, 3)
        gradient_matrix[NUM.arange(n).reshape((-1, 1)), sort_matrix[:, bw:]] = 0

        if index is None:
            return weight_matrix, gradient_matrix
        return weight_matrix[index, :], gradient_matrix[index, :]

    if index is None:
        return weight_matrix
    return weight_matrix[index, :]

########################### create synthetic data ###########################
# Gaussian MGWR can make sure improvement in this case
# bw=86, bw=[30., 182., 311., 400.]
# sum(err2)=44.4, =39.4
def test1():
    NUM.random.seed(1)
    k = 3
    n = 400
    n0 = 20

    xcoord = NUM.tile(range(n0), n0) + NUM.random.random(size=n)
    ycoord = NUM.repeat(range(n0), n0) + NUM.random.random(size=n)
    coords = list(zip(xcoord, ycoord))

    x = NUM.random.random((n, k + 1)) * 3
    x[:,0] = 1

    opt_coef = NUM.ones((n, k+1))
    opt_coef[:, 1] = NUM.sqrt((xcoord - xcoord.mean())**2 + (ycoord - ycoord.mean())**2)
    opt_coef[:, 2] = (xcoord - xcoord.mean())**3/30 + NUM.sqrt(ycoord)*5
    opt_coef[:, 3] = NUM.sin(xcoord*2*NUM.pi/5) * 20 + NUM.cos(ycoord*2*NUM.pi/5) * 20
    # opt_coef.mean(axis=0)
    # NUM.corrcoef(opt_coef.T)

    opt_bw = NUM.array([101, 33, 101, 33])
    opt_coef_weight = NUM.array([1, -1, 1, -1])

    y = NUM.random.random(size=(n,1))
    coef = NUM.zeros((n, k+1))
    for j in range(k+1):
        distance_matrix, weight_matrix = distanceMatrix(coords, bw=opt_bw[j])
        for i in range(n):
            coef[i, j] = NUM.sum(x[:, j] * weight_matrix[i, :]) / opt_bw[j] * opt_coef_weight[j]
            # coef[i, j] = NUM.sum(opt_coef[:, j] * weight_matrix[i, :]) / opt_bw[j] * opt_coef_weight[j]
            y[i] = y[i] + x[i, j] * coef[i, j]

    y = (y - y.mean(axis=0)) / y.std(axis=0)
    x[:, 1:] = (x[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)

    coef.mean(axis=0) # array([ 0.37366147, -2.7463686 ,  5.59265511,  0.29424007])

    # from mpl_toolkits import mplot3d
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), opt_coef[:, 0].reshape((n0, n0)))
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), opt_coef[:, 1].reshape((n0, n0)))
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), opt_coef[:, 2].reshape((n0, n0)))
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), opt_coef[:, 3].reshape((n0, n0)))
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), coef[:, 0].reshape((n0, n0)))
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), coef[:, 1].reshape((n0, n0)))
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), coef[:, 2].reshape((n0, n0)))
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), coef[:, 3].reshape((n0, n0)))

    return (x, y, coords, n, k)

########################### create synthetic data ###########################
# Gaussian MGWR can make sure improvement in this case
# bw=30, bw=[30., 91., 57., 30.]
# aicc=309.9, 180.5
# sum(err2)=18.3, =18.8
def test2():
    NUM.random.seed(1)
    k = 3
    n = 400
    n0 = 20

    xcoord = NUM.tile(range(n0), n0) + NUM.random.random(size=n)
    ycoord = NUM.repeat(range(n0), n0) + NUM.random.random(size=n)
    coords = list(zip(xcoord, ycoord))

    x = NUM.random.random((n, k + 1)) * 3
    x[:,0] = 1

    opt_coef = NUM.ones((n, k+1))
    opt_coef[:, 1] = NUM.sqrt((xcoord - xcoord.mean())**2 + (ycoord - ycoord.mean())**2)
    opt_coef[:, 2] = (xcoord - xcoord.mean())**3/30 + NUM.sqrt(ycoord)*5
    opt_coef[:, 3] = NUM.sin(xcoord*2*NUM.pi/5) * 20 + NUM.cos(ycoord*2*NUM.pi/5) * 20
    # opt_coef.mean(axis=0)
    # NUM.corrcoef(opt_coef.T)

    opt_bw = NUM.array([33, 101, 101, 33])
    opt_coef_weight = NUM.array([1, -1, 1, -1])

    y = NUM.random.random(size=(n,1))
    coef = NUM.zeros((n, k+1))
    for j in range(k+1):
        distance_matrix, weight_matrix = distanceMatrix(coords, bw=opt_bw[j])
        for i in range(n):
            coef[i, j] = NUM.sum(opt_coef[:, j] * weight_matrix[i, :]) / opt_bw[j] * opt_coef_weight[j]
            y[i] = y[i] + x[i, j] * coef[i, j]

    y = (y - y.mean(axis=0)) / y.std(axis=0)
    x[:, 1:] = (x[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)

    coef.mean(axis=0) # array([ 0.35899738, -2.74736397,  5.59265511,  0.29424007])

    # from mpl_toolkits import mplot3d
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), opt_coef[:, 0].reshape((n0, n0)))
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), opt_coef[:, 1].reshape((n0, n0)))
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), opt_coef[:, 2].reshape((n0, n0)))
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), opt_coef[:, 3].reshape((n0, n0)))
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), coef[:, 0].reshape((n0, n0)))
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), coef[:, 1].reshape((n0, n0)))
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), coef[:, 2].reshape((n0, n0)))
    # ax.scatter3D(xcoord.reshape((n0, n0)), ycoord.reshape((n0, n0)), coef[:, 3].reshape((n0, n0)))

    return (x, y, coords, n, k)

def get_covid_data():
    data_path = os.path.join(data_dir, "moddat_Feb2021.csv")
    df = pd.DataFrame.dropna(df)
    df = df.loc[df.month == 8, :]

    depVarName = 'relative_change_feb'
    indVarNames = ['tmmx', 'rmax', 'sph', 'pm25_raw', 'cases', 'day']
    df = df.loc[:, indVarNames + [depVarName, "Long", "Lat"]]

    n = df.shape[0]
    k = len(indVarNames)

    y = df[depVarName].values.reshape((-1,1))
    x = NUM.ones((n, k+1), dtype=float)
    for column, variable in enumerate(indVarNames):
        varData = df[[variable]].values
        x[:, column + 1] = varData.flatten()

    coords = list(zip(df['Long'], df['Lat']))
    return (x, y, coords, n, k)