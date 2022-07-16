import csv
from matplotlib import image
import numpy as np
import cv2 as cv
import scipy.signal
import math
from scipy.signal import find_peaks
from scipy.spatial.distance import directed_hausdorff
import similaritymeasures
np.set_printoptions(precision=3)

#Spectrum function, import image, return spectrum array (not n. size)
def spectrum(img: image) -> np.ndarray:
    spec = np.empty((0))
    h, w, _ = img.shape
    for x in range(0,w):
        [r,g,b]=img[255,x]
        intensity = (int(r)+int(g)+int(b))/3
        spec = np.append(spec, intensity)
    return spec

#Spectrum function, import image, return normal spectrum array
def spectrum_norml(img: image) -> np.ndarray:
    dsize = (1280, 720)
    imgf = cv.resize(img, dsize)
    h, w, _ = imgf.shape
    spec = np.empty((0))
    for x in range(0,w):
        [r,g,b]=imgf[255,x]
        intensity = (int(r)+int(g)+int(b))/3
        spec = np.append(spec, intensity)
    return spec

#Spectrum filtering function, import spec array, return filt. spec array
def spectrum_filtrd(spec: np.ndarray) -> np.ndarray:
    b, a = scipy.signal.butter(10, 0.1)
    filt = scipy.signal.filtfilt(b, a, spec)
    return filt

#All peaks function, import (filt.) spec array a int b int, return 1d all peaks array in range
def all_pksinrange(fspec: np.ndarray, a: int, b: int):
    Apk, _ = find_peaks(fspec, height=(a, b), distance=5)
    return Apk

#High peaks function, import (filt.) spec array, return h. peaks array
def high_peaks(fspec: np.ndarray):
    fHpks = all_pksinrange(fspec, 101, 255)
    return fHpks

#Middle peaks function, import (filt.) spec array, return m. peaks array
def middle_peaks(fspec: np.ndarray):
    fMpks = all_pksinrange(fspec, 41, 100)
    return fMpks

#Low peaks function, import (filt.) spec array, return l. peaks array
def low_peaks(fspec: np.ndarray):
    fLpks = all_pksinrange(fspec, 0, 40)
    return fLpks

#All peaks function, import (filt.) spec array a int b int, return sorted list 2d h. peaks in range
def all_pksinrange_sortd2d(fspec: np.ndarray, a: int, b: int) -> np.ndarray:
    Apk, _ = find_peaks(fspec, height=(a, b), distance=5)
    fApk = np.array(Apk)
    ffApk = np.array(fspec[Apk])
    mApk = np.dstack((fApk, ffApk))
    mmApk = mApk[0]
    fApks = sorted(mmApk,key=lambda x: x[1])
    ffApks = np.array(fApks)
    return ffApks

#Total peaks function, import (filt.) spec array, return sorted list 2d h. peaks
def total_peaks_sortd2d(fspec: np.ndarray) -> np.ndarray:
    fHpks = all_pksinrange_sortd2d(fspec, 0, 255)
    return fHpks

#High peaks function, import (filt.) spec array, return sorted list 2d h. peaks
def high_peaks_sortd2d(fspec: np.ndarray) -> np.ndarray:
    fHpks = all_pksinrange_sortd2d(fspec, 101, 255)
    return fHpks

#Middle peaks function, import (filt.) spec array, return list 2d m. peaks
def middle_peaks_sortd2d(fspec: np.ndarray) -> np.ndarray:
    fMpks = all_pksinrange_sortd2d(fspec, 41, 100)
    return fMpks

#Low peaks function, import (filt.) spec array, return list 2d l. peaks
def low_peaks_sortd2d(fspec: np.ndarray) -> np.ndarray:
    fLpks = all_pksinrange_sortd2d(fspec, 0, 40)
    return fLpks

#Difference function, import two n. arrays, return diff. float
def arrays_vertcl_diff(d1: np.ndarray, d2: np.ndarray) -> float:
    csvl = 1280
    df = np.empty(0)
    for i in d1:
        difi = abs(d2[i] - d1[i])
        df = np.append(df, difi)
    ttl = np.sum(df)
    dttl = (ttl/csvl)*(100/255)
    return dttl

#Matrix horizontal distance function, import list 2d peaks, return matrix
def matrix_horztl_dist(fnHpk: np.ndarray) -> np.ndarray:
    nmbpks = len(fnHpk)
    ffnHpk = fnHpk[::-1]
    XnHpk = [x for x,y in ffnHpk]
    mtrx = np.zeros(shape=(nmbpks,nmbpks))
    for i in range(nmbpks):
        for j in range(nmbpks):
            mtrx[i][j] = abs(XnHpk[i] - XnHpk[j])
    return mtrx

#Matrix vertical distance function, import list 2d peaks, return matrix
def matrix_vertcl_dist(fnHpk: np.ndarray) -> np.ndarray:
    nmbpks = len(fnHpk)
    ffnHpk = fnHpk[::-1]
    YnHpk = [y for x,y in ffnHpk]
    mtrx = np.zeros(shape=(nmbpks,nmbpks))
    for i in range(nmbpks):
        for j in range(nmbpks):
            mtrx[i][j] = abs(YnHpk[i] - YnHpk[j])
    return mtrx

#Matrix pythagorean distance function, import list 2d peaks, return matrix
def matrix_pytgrn_dist(fnHpk: np.ndarray) -> np.ndarray:
    nmbpks = len(fnHpk)
    ffnHpk = fnHpk[::-1]
    YnHpk = [y for x,y in ffnHpk]
    XnHpk = [x for x,y in ffnHpk]
    mtrx = np.zeros(shape=(nmbpks,nmbpks))
    for i in range(nmbpks):
        for j in range(nmbpks):
            mtrx[i][j] = math.sqrt((XnHpk[i] - XnHpk[j])**2 + (YnHpk[i] - YnHpk[j])**2)
    return mtrx

#Matrixs difference, import 2 square matrx, return difference float
def matrixs_diffrnc (matrx1: np.ndarray, matrx2: np.ndarray) -> float:
    nmtrx1 = matrx1.shape[1]
    nmtrx2 = matrx2.shape[1]
    f = min(nmtrx1, nmtrx2)
    mtrx1 = matrx1[0:f, 0:f]
    mtrx2 = matrx2[0:f, 0:f]
    nmtrx = abs(mtrx1 - mtrx2)
    tdiff = np.sum(nmtrx) / f**2
    return tdiff

#1dim interpolation f, import fspec array n int, return stretched spec of n length
def interpltn_1dim(fspec: np.ndarray, n: int) -> np.ndarray:
    lspc = len(fspec)
    fspecf = np.interp(np.linspace(0, lspc - 1, num=n), np.arange(lspc), fspec)
    return fspecf

#horizontal right multiplication f, import fspec array ini int end int n float, return spec with stretched interval
def multiply_horztl_right(fspec: np.ndarray, ini: int, end: int, n: int) -> np.ndarray:
    lspc = len(fspec)
    if abs(end-ini)==abs(n) or abs(end-ini)==0 or n<=0 or end<ini or abs(n)+abs(ini) > lspc:
        return fspec
    else:
        m = abs(lspc - (ini + abs(n)))
        specm = interpltn_1dim(np.array(fspec[ini:end]), int(n))
        specl = interpltn_1dim(np.array(fspec[end:]), int(m))
        specf = np.concatenate((fspec[:ini], specm, specl))
        return specf

#horizontal left multiplication f, import fspec array ini int end int n float, return spec with stretched interval
def multiply_horztl_left(fspec: np.ndarray, ini: int, end: int, n: int) -> np.ndarray:
    lspc = len(fspec)
    if abs(end-ini)== abs(n) or abs(end-ini)==0 or n<=0 or end<ini or n+end>lspc or end<n:
        return fspec
    else:
        m = abs(end - abs(n))
        speci = interpltn_1dim(np.array(fspec[:ini]), int(m))
        specm = interpltn_1dim(np.array(fspec[ini:end]), int(n))
        specf = np.concatenate((speci, specm, fspec[end:]))
        return specf

#vertical multiplication f, import fspec array ini int end int n float, return spec with stretched interval
def multiply_vertcl(fspec: np.ndarray, ini: int, end: int, n: float) -> np.ndarray:
    specm = np.multiply(fspec[ini:end], n)
    specf = np.concatenate((fspec[:ini],specm,fspec[end:]))
    return specf

#codeword function, import fspec array, return cdwd string
def codeword_numbr(fspec: np.ndarray) -> str:
    darrpks = all_pksinrange_sortd2d(fspec, 0, 255)[::-1]
    mtdarrpks = darrpks[0:10]
    nnmbdt = list(range(0,10))
    ntdmtrx = [(*i, j) for i, j in zip(mtdarrpks, nnmbdt)]
    natdmtrx = np.asarray(ntdmtrx)
    nstdmtrx = natdmtrx[natdmtrx[:, 0].argsort()]
    cdwd = ""
    nwdmtrx = nstdmtrx[:,2]
    for i in nwdmtrx:
        cdwd += str(int(i))
    return cdwd

#codeword matching f, import 2 arrays, return 1 int (from 1 to 9)
def codeword_matchn(fspec1: np.ndarray, fspec2: np.ndarray) -> int:
    if codeword_numbr(fspec1, 9) == codeword_numbr(fspec2, 9):
        return 9
    elif codeword_numbr(fspec1, 8) == codeword_numbr(fspec2, 8):
        return 8
    elif codeword_numbr(fspec1, 7) == codeword_numbr(fspec2, 7):
        return 7
    elif codeword_numbr(fspec1, 6) == codeword_numbr(fspec2, 6):
        return 6
    elif codeword_numbr(fspec1, 5) == codeword_numbr(fspec2, 5):
        return 5
    elif codeword_numbr(fspec1, 4) == codeword_numbr(fspec2, 4):
        return 4
    elif codeword_numbr(fspec1, 3) == codeword_numbr(fspec2, 3):
        return 3
    elif codeword_numbr(fspec1, 2) == codeword_numbr(fspec2, 2):
        return 2
    elif codeword_numbr(fspec1, 1) == codeword_numbr(fspec2, 1):
        return 1

#codeword f, import fspec array, return cdwd string
def codeword_pksnumbr(fspec: np.ndarray) -> str:
    lHpks = len(high_peaks(fspec))
    lMpks = len(middle_peaks(fspec))
    lLpks = len(low_peaks(fspec))
    cdwd = "H"+str(lHpks)+"M"+str(lMpks)+"L"+str(lLpks)
    return cdwd

#csv to array f, import csv, return array
def csv_to_array(csvdt: csv) -> np.ndarray:
    spec = np.empty(0)
    for i in csvdt:
        dif = csvdt[i]
        extrc_spec = np.append(spec,dif)
    return extrc_spec

#distance and area functions for curves (arrays)
#directed hausdorff distance f, import 2 spec, return float
def directed_hausdorff_distance(spec1: np.ndarray, spec2: np.ndarray) -> float:
    x = list(range(0, len(spec1)))
    y = spec1
    P = np.array([x, y]).T
    z = list(range(0, len(spec2)))
    w = spec2
    Q = np.array([z, w]).T
    dh, ind1, ind2 = directed_hausdorff(P, Q)
    return dh

#discrete frechet distance f, import 2 spec, return float
def discrete_frechet_distance(spec1: np.ndarray, spec2: np.ndarray) -> float:
    x = list(range(0, len(spec1)))
    y = spec1
    P = np.array([x, y]).T
    z = list(range(0, len(spec2)))
    w = spec2
    Q = np.array([z, w]).T
    df = similaritymeasures.frechet_dist(P, Q)
    return df

#dynamic time warping f, import 2 spec, return float
def dynamic_time_warping(spec1: np.ndarray, spec2: np.ndarray) -> float:
    x = list(range(0, len(spec1)))
    y = spec1
    P = np.array([x, y]).T
    z = list(range(0, len(spec2)))
    w = spec2
    Q = np.array([z, w]).T
    dtw, d = similaritymeasures.dtw(P, Q)
    return dtw

#partial curve mapping f, import 2 spec, return float
def partial_curve_mapping(spec1: np.ndarray, spec2: np.ndarray) -> float:
    x = list(range(0, len(spec1)))
    y = spec1
    P = np.array([x, y]).T
    z = list(range(0, len(spec2)))
    w = spec2
    Q = np.array([z, w]).T
    pcm = similaritymeasures.pcm(P, Q)
    return pcm

#area between curves f, import 2 spec, return float
def area_between_curves(spec1: np.ndarray, spec2: np.ndarray) -> float:
    x = list(range(0, len(spec1)))
    y = spec1
    P = np.array([x, y]).T
    z = list(range(0, len(spec2)))
    w = spec2
    Q = np.array([z, w]).T
    area = similaritymeasures.area_between_two_curves(P, Q)
    return area

#curvelength distance metric f, import 2 spec, return float
def curvelength_distance_metric(spec1: np.ndarray, spec2: np.ndarray) -> float:
    x = list(range(0, len(spec1)))
    y = spec1
    P = np.array([x, y]).T
    z = list(range(0, len(spec2)))
    w = spec2
    Q = np.array([z, w]).T
    cl = similaritymeasures.curve_length_measure(P, Q)
    return cl

#normalizer 1 iteration of specs, import 2 spec, return spec, makes biggest peaks match
#aim of making spec2 as similar as spec1, then one could consider dtw and check the dist
#of a base spec, in case they don't match, there is no purpose in continuing the search
def normalizer1it_spec(spec1: np.ndarray, spec2: np.ndarray) -> np.ndarray:
    mtrx1 = all_pksinrange_sortd2d(spec1, 0, 255)[::-1]
    mtrx2 = all_pksinrange_sortd2d(spec2, 0, 255)[::-1]
    x = 0
    y = 1
    nntrx1 = mtrx1[0]
    nntrx2 = mtrx2[0]
    xnspec2 = multiply_vertcl(spec2, 0, 1280, float(nntrx1[y] / nntrx2[y]))
    nspec2 = multiply_horztl_right(xnspec2, 0, int(nntrx2[x]), int(nntrx1[x]))
    return nspec2

#normalizer of specs ver 1.0, still in development, takes 2 arrays, spec1 and spec2, n number is given
#by matching codeword f, has the purpose of checking the 9 highest peaks and match them in order acc
#to the specs, make peaks from spec2 similar to 1, requires additional functions, also it was not
#possible to index most of the operations due the nature of the indexes, 'mtrx2[0][x]'
def normalizer_spec(spec1: np.ndarray, spec2: np.ndarray, nm: int):
    mtrx1 = all_pksinrange_sortd2d(spec1, 0, 255)[::-1]
    mtrx2 = all_pksinrange_sortd2d(spec2, 0, 255)[::-1]
    x = 0
    y = 1
    num = min(mtrx1.shape[0] , mtrx2.shape[0])
    nitrx1 = mtrx1[0:num]
    nitrx2 = mtrx2[0:num]
    nmtrx1 = nitrx1[0:nm]
    nmtrx2 = nitrx2[0:nm]
    nntrx1 = nmtrx1[nmtrx1[:, 0].argsort()]
    nntrx2 = nmtrx2[nmtrx2[:, 0].argsort()]
    if nm==1:
        xnspec = multiply_vertcl(spec2, 0, 1280, float(mtrx1[0][y] / mtrx2[0][y]))
        nspec = multiply_horztl_right(xnspec, 0, int(mtrx2[0][x]), int(mtrx1[0][x]))
        return nspec
    elif nm==2:
        xnspec1 = interpltn_1dim(spec2[0:int(nntrx2[0][x])], int(nntrx1[0][x]))
        xnspec2 = interpltn_1dim(spec2[int(nntrx2[0][x]):int(nntrx2[1][x])], int(abs(nntrx1[1][x] - nntrx1[0][x])))
        xnspec3 = interpltn_1dim(spec2[int(nntrx2[1][x]):1280], 1280 - int(nntrx1[1][x]))
        xnspec = np.concatenate((xnspec1, xnspec2, xnspec3))
        nspec = multiply_vertcl(xnspec, 0, 1280, float(nntrx1[0][y] / nntrx2[0][y]))
        return nspec
    elif nm==3:
        xnspec1 = interpltn_1dim(spec2[0:int(nntrx2[0][x])], int(nntrx1[0][x]))
        xnspec2 = interpltn_1dim(spec2[int(nntrx2[0][x]):int(nntrx2[1][x])], int(abs(nntrx1[1][x] - nntrx1[0][x])))
        xnspec3 = interpltn_1dim(spec2[int(nntrx2[1][x]):int(nntrx2[2][x])], int(abs(nntrx1[2][x] - nntrx1[1][x])))
        xnspec4 = interpltn_1dim(spec2[int(nntrx2[2][x]):1280], 1280 - int(nntrx1[2][x]))
        xnspec = np.concatenate((xnspec1, xnspec2, xnspec3, xnspec4))
        nspec = multiply_vertcl(xnspec, 0, 1280, float(mtrx1[0][y] / mtrx2[0][y]))
        return nspec
    elif nm==4:
        xnspec1 = interpltn_1dim(spec2[0:int(nntrx2[0][x])], int(nntrx1[0][x]))
        xnspec2 = interpltn_1dim(spec2[int(nntrx2[0][x]):int(nntrx2[1][x])], int(abs(nntrx1[1][x] - nntrx1[0][x])))
        xnspec3 = interpltn_1dim(spec2[int(nntrx2[1][x]):int(nntrx2[2][x])], int(abs(nntrx1[2][x] - nntrx1[1][x])))
        xnspec4 = interpltn_1dim(spec2[int(nntrx2[2][x]):int(nntrx2[3][x])], int(abs(nntrx1[3][x] - nntrx1[2][x])))
        xnspec5 = interpltn_1dim(spec2[int(nntrx2[3][x]):1280], 1280 - int(nntrx1[3][x]))
        xnspec = np.concatenate((xnspec1, xnspec2, xnspec3, xnspec4, xnspec5))
        nspec = multiply_vertcl(xnspec, 0, 1280, float(mtrx1[0][y] / mtrx2[0][y]))
        return nspec
    elif nm==5:
        xnspec1 = interpltn_1dim(spec2[0:int(nntrx2[0][x])], int(nntrx1[0][x]))
        xnspec2 = interpltn_1dim(spec2[int(nntrx2[0][x]):int(nntrx2[1][x])], int(abs(nntrx1[1][x] - nntrx1[0][x])))
        xnspec3 = interpltn_1dim(spec2[int(nntrx2[1][x]):int(nntrx2[2][x])], int(abs(nntrx1[2][x] - nntrx1[1][x])))
        xnspec4 = interpltn_1dim(spec2[int(nntrx2[2][x]):int(nntrx2[3][x])], int(abs(nntrx1[3][x] - nntrx1[2][x])))
        xnspec5 = interpltn_1dim(spec2[int(nntrx2[3][x]):int(nntrx2[4][x])], int(abs(nntrx1[4][x] - nntrx1[3][x])))
        xnspec6 = interpltn_1dim(spec2[int(nntrx2[4][x]):1280], 1280 - int(nntrx1[4][x]))
        xnspec = np.concatenate((xnspec1, xnspec2, xnspec3, xnspec4, xnspec5, xnspec6))
        nspec = multiply_vertcl(xnspec, 0, 1280, float(mtrx1[0][y] / mtrx2[0][y]))
        return nspec
    elif nm==6:
        xnspec1 = interpltn_1dim(spec2[0:int(nntrx2[0][x])], int(nntrx1[0][x]))
        xnspec2 = interpltn_1dim(spec2[int(nntrx2[0][x]):int(nntrx2[1][x])], int(abs(nntrx1[1][x] - nntrx1[0][x])))
        xnspec3 = interpltn_1dim(spec2[int(nntrx2[1][x]):int(nntrx2[2][x])], int(abs(nntrx1[2][x] - nntrx1[1][x])))
        xnspec4 = interpltn_1dim(spec2[int(nntrx2[2][x]):int(nntrx2[3][x])], int(abs(nntrx1[3][x] - nntrx1[2][x])))
        xnspec5 = interpltn_1dim(spec2[int(nntrx2[3][x]):int(nntrx2[4][x])], int(abs(nntrx1[4][x] - nntrx1[3][x])))
        xnspec6 = interpltn_1dim(spec2[int(nntrx2[4][x]):int(nntrx2[5][x])], int(abs(nntrx1[5][x] - nntrx1[4][x])))
        xnspec7 = interpltn_1dim(spec2[int(nntrx2[5][x]):1280], 1280 - int(nntrx1[5][x]))
        xnspec = np.concatenate((xnspec1, xnspec2, xnspec3, xnspec4, xnspec5, xnspec6, xnspec7))
        nspec = multiply_vertcl(xnspec, 0, 1280, float(mtrx1[0][y] / mtrx2[0][y]))
        return nspec
    elif nm==7:
        xnspec1 = interpltn_1dim(spec2[0:int(nntrx2[0][x])], int(nntrx1[0][x]))
        xnspec2 = interpltn_1dim(spec2[int(nntrx2[0][x]):int(nntrx2[1][x])], int(abs(nntrx1[1][x] - nntrx1[0][x])))
        xnspec3 = interpltn_1dim(spec2[int(nntrx2[1][x]):int(nntrx2[2][x])], int(abs(nntrx1[2][x] - nntrx1[1][x])))
        xnspec4 = interpltn_1dim(spec2[int(nntrx2[2][x]):int(nntrx2[3][x])], int(abs(nntrx1[3][x] - nntrx1[2][x])))
        xnspec5 = interpltn_1dim(spec2[int(nntrx2[3][x]):int(nntrx2[4][x])], int(abs(nntrx1[4][x] - nntrx1[3][x])))
        xnspec6 = interpltn_1dim(spec2[int(nntrx2[4][x]):int(nntrx2[5][x])], int(abs(nntrx1[5][x] - nntrx1[4][x])))
        xnspec7 = interpltn_1dim(spec2[int(nntrx2[5][x]):int(nntrx2[6][x])], int(abs(nntrx1[6][x] - nntrx1[5][x])))       
        xnspec8 = interpltn_1dim(spec2[int(nntrx2[6][x]):1280], 1280 - int(nntrx1[6][x]))
        xnspec = np.concatenate((xnspec1, xnspec2, xnspec3, xnspec4, xnspec5, xnspec6, xnspec7, xnspec8))
        nspec = multiply_vertcl(xnspec, 0, 1280, float(mtrx1[0][y] / mtrx2[0][y]))
        return nspec
    elif nm==8:
        xnspec1 = interpltn_1dim(spec2[0:int(nntrx2[0][x])], int(nntrx1[0][x]))
        xnspec2 = interpltn_1dim(spec2[int(nntrx2[0][x]):int(nntrx2[1][x])], int(abs(nntrx1[1][x] - nntrx1[0][x])))
        xnspec3 = interpltn_1dim(spec2[int(nntrx2[1][x]):int(nntrx2[2][x])], int(abs(nntrx1[2][x] - nntrx1[1][x])))
        xnspec4 = interpltn_1dim(spec2[int(nntrx2[2][x]):int(nntrx2[3][x])], int(abs(nntrx1[3][x] - nntrx1[2][x])))
        xnspec5 = interpltn_1dim(spec2[int(nntrx2[3][x]):int(nntrx2[4][x])], int(abs(nntrx1[4][x] - nntrx1[3][x])))
        xnspec6 = interpltn_1dim(spec2[int(nntrx2[4][x]):int(nntrx2[5][x])], int(abs(nntrx1[5][x] - nntrx1[4][x])))
        xnspec7 = interpltn_1dim(spec2[int(nntrx2[5][x]):int(nntrx2[6][x])], int(abs(nntrx1[6][x] - nntrx1[5][x])))       
        xnspec8 = interpltn_1dim(spec2[int(nntrx2[6][x]):int(nntrx2[7][x])], int(abs(nntrx1[7][x] - nntrx1[6][x])))
        xnspec9 = interpltn_1dim(spec2[int(nntrx2[7][x]):1280], 1280 - int(nntrx1[7][x]))
        xnspec = np.concatenate((xnspec1, xnspec2, xnspec3, xnspec4, xnspec5, xnspec6, xnspec7, xnspec8, xnspec9))
        nspec = multiply_vertcl(xnspec, 0, 1280, float(mtrx1[0][y] / mtrx2[0][y]))
        return nspec
    elif nm==9:
        xnspec1 = interpltn_1dim(spec2[0:int(nntrx2[0][x])], int(nntrx1[0][x]))
        xnspec2 = interpltn_1dim(spec2[int(nntrx2[0][x]):int(nntrx2[1][x])], int(abs(nntrx1[1][x] - nntrx1[0][x])))
        xnspec3 = interpltn_1dim(spec2[int(nntrx2[1][x]):int(nntrx2[2][x])], int(abs(nntrx1[2][x] - nntrx1[1][x])))
        xnspec4 = interpltn_1dim(spec2[int(nntrx2[2][x]):int(nntrx2[3][x])], int(abs(nntrx1[3][x] - nntrx1[2][x])))
        xnspec5 = interpltn_1dim(spec2[int(nntrx2[3][x]):int(nntrx2[4][x])], int(abs(nntrx1[4][x] - nntrx1[3][x])))
        xnspec6 = interpltn_1dim(spec2[int(nntrx2[4][x]):int(nntrx2[5][x])], int(abs(nntrx1[5][x] - nntrx1[4][x])))
        xnspec7 = interpltn_1dim(spec2[int(nntrx2[5][x]):int(nntrx2[6][x])], int(abs(nntrx1[6][x] - nntrx1[5][x])))       
        xnspec8 = interpltn_1dim(spec2[int(nntrx2[6][x]):int(nntrx2[7][x])], int(abs(nntrx1[7][x] - nntrx1[6][x])))
        xnspec9 = interpltn_1dim(spec2[int(nntrx2[7][x]):int(nntrx2[8][x])], int(abs(nntrx1[8][x] - nntrx1[7][x])))
        xnspec10 = interpltn_1dim(spec2[int(nntrx2[8][x]):1280], 1280 - int(nntrx1[8][x]))
        xnspec = np.concatenate((xnspec1, xnspec2, xnspec3, xnspec4, xnspec5, xnspec6, xnspec7, xnspec8, xnspec9, xnspec10))
        nspec = multiply_vertcl(xnspec, 0, 1280, float(mtrx1[0][y] / mtrx2[0][y]))
        return nspec

    