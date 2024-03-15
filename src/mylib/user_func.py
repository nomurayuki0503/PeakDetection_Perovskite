import numpy as np
import scipy.optimize as opt
from scipy.ndimage import maximum_filter
from matplotlib.colors import LinearSegmentedColormap
import pymc3 as pm
import arviz as az
from sklearn.cluster import KMeans

def detect_peaks(data, edge, filter_size=3, order=0.5):
    data_ = data.copy()
    data_[0:int(filter_size*edge), :] = 0
    data_[-int(filter_size*edge):, :] = 0
    data_[:, 0:int(filter_size*edge)] = 0
    data_[:, -int(filter_size*edge):] = 0
    local_max = maximum_filter(data_, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(data_, mask=~(data_ == local_max))

    # Remove small peaks
    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max()*order))
    peaks_index = np.where((temp.mask != True))
    return peaks_index

# Define gaussian function
def twoD_Gaussian(XY, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x,y = XY[0:2]
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

# Calculate cetroid
def centroid(data):
    h,w = np.shape(data)
    x = np.arange(0,w)
    y = np.arange(0,h)
    X,Y = np.meshgrid(x,y)
    cx = np,sum(X*data)/np,sum(data)
    cy = np,sum(Y*data)/np,sum(data)
    return cx, cy

# Fit gaussian
def gauss(data, size_peak, FWHM):
    # Create x and y indices
    x_i = np.linspace(0, size_peak-1, size_peak)
    y_i = np.linspace(0, size_peak-1, size_peak)
    x_i, y_i = np.meshgrid(x_i, y_i)
    data = np.reshape(data, (size_peak*size_peak))
    popt, pcov = opt.curve_fit(twoD_Gaussian, (x_i, y_i), data, p0=(max(data)-min(data), size_peak/2, size_peak/2, FWHM/2.35, FWHM/2.35, 1, min(data)))
    #intens = popt[0]
    cx = popt[1]
    cy = popt[2]
    data_fitted = twoD_Gaussian((x_i, y_i), *popt)
    return cx, cy
    #return intens, cx, cy

# Bayesian Inference
def bayesian_inference(size_peak, patch, num_sample, chains):

    # Make x-y grating
    x_linspace = np.linspace(0, size_peak-1, size_peak)
    y_linspace = np.linspace(0, size_peak-1, size_peak)
    x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)
    x_coords, y_coords = x_grid.ravel(), y_grid.ravel()
    xy_coords = np.stack((x_coords, y_coords), axis=0).T    #pymc3の構造に合わせて転置

    # Define model for gaussian fitting
    with pm.Model() as model:
        xy = pm.Data('Coordinate', xy_coords)
        mu = pm.Uniform('Atom Position (μ)', lower=0, upper=size_peak, shape=2)
        sigma = pm.Uniform('Noise (σ)', lower=0, upper=np.max(patch.ravel()))
        bckgrd = pm.Uniform('Background Intensity', lower=0, upper=np.mean(patch.ravel()))
        peak = pm.Uniform('Peak Intensity', lower=0, upper=np.max(patch.ravel())*2)
        sd_dist = pm.Exponential.dist(1)
        packed_chol = pm.LKJCholeskyCov("Cholesky Matrix", n=2, eta=1, sd_dist=sd_dist, compute_corr=False)
        chol = pm.expand_packed_triangular(2, packed_chol, lower=True)
        cov = pm.Deterministic('Covariance Matrix', chol.dot(chol.T))
        inv =  pm.math.matrix_inverse(cov)
        y = pm.Normal('Intensity', mu=(np.exp( -pm.math.extract_diag(pm.math.matrix_dot((xy-mu), inv, (xy-mu).T))/2.0)) * peak + bckgrd, sigma=sigma, observed=patch.ravel())

    # MCMC Sample
    with model:
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=chains,
            return_inferencedata=True
        )

    # Sample atomic positions from posterior distribution
    with model:
        pred_atom_center = pm.sample_posterior_predictive(
            trace, 
            samples=num_sample, 
            var_names=['Atom Position (μ)']
        )

    return az.summary(trace), pred_atom_center

# Make color map
def generate_cmap(colors):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for v,c in zip(values, colors):
        color_list.append((v/vmax, c))
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)

# Analysis for SrTiO3-like pervskite materials
#def analysis_pervskite(Z, col, peaks_x, peaks_y, sd_x_bayes, sd_y_bayes, size_peak, fit_method):
def analysis_pervskite(Z, col, peaks_x, peaks_y, size_peak):
    # Clustering
    kmeans_model = KMeans(n_clusters=col).fit(np.reshape(peaks_y, [-1, 1]))    # Kmeans法を用いて輝点を行ごとに分類(y座標の値で判定)
    labels = kmeans_model.labels_
    cluster_centers = kmeans_model.cluster_centers_                            # 各行の原子カラムのy座標の代表値を"cluster_centers"に格納。"cluster_centers"の要素数は行数と同じ
    cluster_centers = np.array([ e for row in cluster_centers for e in row ])  # "cluster_centers"を1次元ベクトルに展開

    # Align factors
    cluster_index = { k:v for k,v in enumerate(cluster_centers.argsort()) }    # "cluster_index"は"cluster_centers"の各要素が、何番目に小さいかを示している。「0:8」であれば、"cluster_centers"の要素0は9番目に小さい(下から9行目の原子カラム列)
    align_peaks_x = np.array( [ np.array(peaks_x)[np.where(labels == cluster_index[h])] for h in range(col) ], dtype='object' )         # "peaks_x"の値を小さい順に各行ごとに取り出してndarray化
    align_peaks_y = np.array( [ np.array(peaks_y)[np.where(labels == cluster_index[h])] for h in range(col) ], dtype='object' )         # "peaks_y"の値を小さい順に各行ごとに取り出してndarray化
    x_index = [ { k:v for k,v in enumerate(align_peaks_x[h].argsort()) } for h in range(col) ]                          # 各行内の原子カラムを小さい順に並べるために各要素が何番目に小さいかを辞書化(上と同じことを各行で実施)。
    align_peaks_x =  [ [ align_peaks_x[h][x_index[h][i]] for i in range(len(align_peaks_x[h])) ] for h in range(col) ]  # 各行の"peaks_x"の値を小さい順に取り出してndarray化
    align_peaks_y =  [ [ align_peaks_y[h][x_index[h][i]] for i in range(len(align_peaks_x[h])) ] for h in range(col) ]  # 各行の"peaks_x"の値を小さい順に取り出してndarray化

    # ここまででピーク座標を縦横小さい順に並べ替えた
    flat_align_peaks_x = [ e for row in align_peaks_x for e in row ]
    flat_align_peaks_y = [ e for row in align_peaks_y for e in row ]

    # Define bottom row as Sr aor Ti
    row_1 = [ np.average(Z[int(y-size_peak/2):int(y+size_peak/2), int(x-size_peak/2):int(x+size_peak/2)]) for (x,y) in zip(align_peaks_x[0], align_peaks_y[0]) ]    #1番下の行の原子カラムのsize_peak四方のピクセル強度の平均を取得
    row_2 = [ np.average(Z[int(y-size_peak/2):int(y+size_peak/2), int(x-size_peak/2):int(x+size_peak/2)]) for (x,y) in zip(align_peaks_x[1], align_peaks_y[1]) ]    #下から2行目の原子カラムのsize_peak四方ののピクセル強度の平均を取得

    ave_row_1 = sum(row_1)/len(row_1)              # 1番下の行の原子カラム中心の平均強度を取得
    ave_row_2 = sum(row_2)/len(row_2)              # 下から2番目の行の原子カラム中心の平均強度を取得
    if ave_row_1 >= ave_row_2: bottom_row = "Sr"    # 一番下の原子カラムの行がSr
    if ave_row_1 < ave_row_2: bottom_row = "Ti"    # 一番下の原子カラムの行がTi

    # Define left column as Sr or Ti
    if bottom_row == "Sr":
        if align_peaks_x[0][0] < align_peaks_x[1][0]:
            left_col = "Sr"
        else:
            left_col = "Ti"

    if bottom_row == "Ti":
        if align_peaks_x[0][0] < align_peaks_x[1][0]:
            left_col = "Ti"
        else:
            left_col = "Sr"

    # Remove outliers
    lst = [ len(align_peaks_x[i]) for i in range(col) ]
    lst_even = [ len(align_peaks_x[int(2*i)]) for i in range(int(col/2+col%2)) ]    # 偶数行の原子カラム数を取得(「%」は余り)
    lst_odd = [ len(align_peaks_x[int(2*i+1)]) for i in range(int(col/2)) ]         # 奇数行の原子カラム数を取得
    min_lst_even = min(lst_even)    #偶数行の最小の原子カラム数を取得
    min_lst_odd = min(lst_odd)      #奇数行の最小の原子カラム数を取得

    if bottom_row == "Sr":
        index_even = [ 2*i for i,x in enumerate(lst_even) if x != min_lst_even ]    # 偶数行に対して、原子カラム数が他の行より多い行のインデックスを取得
        index_odd = [ 2*i+1 for i,x in enumerate(lst_odd) if x != min_lst_even-1 ]  # 奇数行に対して、原子カラム数が(偶数行の最小-1)より多い行のインデックスを取得
        dif_even = [ lst[i] - min_lst_even for i in index_even ]                    # 偶数行の原子カラムが他の行より多い行に関して、余分な原子カラム数を取得
        dif_odd = [ lst[i] - (min_lst_even-1) for i in index_odd ]                  # 奇数行の原子カラムが(偶数行の最小-1)より多い行に関して、余分な原子カラム数を取得
        min_index_even = np.argmin(np.array(lst_even))*2                            # 偶数行の中で最小の原子カラム数をもつ行のインデックスを取得
        for i,x in enumerate(index_even):
            for j in range(dif_even[i]):
                if abs(align_peaks_x[x][0]-align_peaks_x[min_index_even][0]) > abs(align_peaks_x[x][-1]-align_peaks_x[min_index_even][-1]):    # 余分な原子カラムが右か左かどちらに出ているかを判定。この条件は左に出ている場合
                    del align_peaks_x[x][0]                                         # 左端の原子カラムの座標を削除
                    del align_peaks_y[x][0]                                         # 左端の原子カラムの座標を削除
                elif abs(align_peaks_x[x][0]-align_peaks_x[min_index_even][0]) < abs(align_peaks_x[x][-1]-align_peaks_x[min_index_even][-1]):  # 余分な原子カラムが右か左かどちらに出ているかを判定。この条件は右に出ている場合
                    del align_peaks_x[x][-1]                                        # 右端の原子カラムの座標を削除
                    del align_peaks_y[x][-1]                                        # 右端の原子カラムの座標を削除
        for i,x in enumerate(index_odd):
            for j in range(dif_odd[i]):
                if align_peaks_x[x][0] < align_peaks_x[min_index_even][0]:          # 余分な原子カラムが右か左かどちらに出ているかを判定。この条件は左に出ている場合
                    del align_peaks_x[x][0]                                         # 左端の原子カラムの座標を削除
                    del align_peaks_y[x][0]                                         # 左端の原子カラムの座標を削除
                elif align_peaks_x[x][-1] > align_peaks_x[min_index_even][-1]:      # 余分な原子カラムが右か左かどちらに出ているかを判定。この条件は右に出ている場合
                    del align_peaks_x[x][-1]                                        # 右端の原子カラムの座標を削除
                    del align_peaks_y[x][-1]                                        # 右端の原子カラムの座標を削除
    elif bottom_row == "Ti":
        index_even = [ 2*i for i,x in enumerate(lst_even) if x != min_lst_odd-1 ]   # 偶数行に対して、原子カラム数が(奇数行の最小-1)より多い行のインデックスを取得
        index_odd = [ 2*i+1 for i,x in enumerate(lst_odd) if x != min_lst_odd ]     # 奇数行に対して、原子カラム数が他の行より多い行のインデックスを取得
        dif_even = [ lst[i] - (min_lst_even-1) for i in index_even ]                # 偶数行の原子カラムが(奇数行の最小-1)より多い行に関して、余分な原子カラム数を取得
        dif_odd = [ lst[i] - min_lst_even for i in index_odd ]                      # 奇数行の原子カラムが他の行より多い行に関して、余分な原子カラム数を取得
        min_index_odd = np.argmin(np.array(lst_odd))*2+1                            # 奇数行の中で最小の原子カラム数をもつ行のインデックスを取得
        for i,x in enumerate(index_even):
            for j in range(dif_even[i]):
                if align_peaks_x[x][0] < align_peaks_x[min_index_odd][0]:           # 余分な原子カラムが右か左かどちらに出ているかを判定。この条件は左に出ている場合
                    del align_peaks_x[x][0]                                         # 左端の原子カラムの座標を削除
                    del align_peaks_y[x][0]                                         # 左端の原子カラムの座標を削除
                elif align_peaks_x[x][-1] > align_peaks_x[min_index_odd][-1]:       # 余分な原子カラムが右か左かどちらに出ているかを判定。左に出ている場合以下を実行
                    del align_peaks_x[x][-1]                                        # 右端の原子カラムの座標を削除
                    del align_peaks_y[x][-1]                                        # 右端の原子カラムの座標を削除
        for i,x in enumerate(index_odd):
            for j in range(dif_odd[i]):
                if abs(align_peaks_x[x][0]-align_peaks_x[min_index_odd][0]) > abs(align_peaks_x[x][-1]-align_peaks_x[min_index_odd][-1]):    # 余分な原子カラムが右か左かどちらに出ているかを判定。この条件は左に出ている場合
                    del align_peaks_x[x][0]                                         # 左端の原子カラムの座標を削除
                    del align_peaks_y[x][0]                                         # 左端の原子カラムの座標を削除
                elif abs(align_peaks_x[x][0]-align_peaks_x[min_index_odd][0]) < abs(align_peaks_x[x][-1]-align_peaks_x[min_index_odd][-1]):  # 余分な原子カラムが右か左かどちらに出ているかを判定。左に出ている場合以下を実行
                    del align_peaks_x[x][-1]                                        # 右端の原子カラムの座標を削除
                    del align_peaks_y[x][-1]                                        # 右端の原子カラムの座標を削除

    # Calculate atomic displacement
    if bottom_row == "Sr":
        if left_col == "Sr":
            center_x = np.array([ [ (align_peaks_x[h*2][i]+align_peaks_x[h*2][i+1]+align_peaks_x[h*2+2][i]+align_peaks_x[h*2+2][i+1])/4 for i in range(len(align_peaks_x[h*2])-1) ] for h in range(int(col/2-1+col%2)) ])    # x座標の重心の計算
            center_y = np.array([ [ (align_peaks_y[h*2][i]+align_peaks_y[h*2][i+1]+align_peaks_y[h*2+2][i]+align_peaks_y[h*2+2][i+1])/4 for i in range(len(align_peaks_x[h*2])-1) ] for h in range(int(col/2-1+col%2)) ])    # y座標の重心の計算
            distance_x = [ align_peaks_x[i*2+1] for i in range(int(col/2-1+col%2)) ] - center_x    # x座標に関して、Srカラムの重心とTiカラムの差分を計算 
            distance_y = [ align_peaks_y[i*2+1] for i in range(int(col/2-1+col%2)) ] - center_y    # y座標に関して、Srカラムの重心とTiカラムの差分を計算
        elif left_col == "Ti":
            center_x = np.array([ [ (align_peaks_x[h*2][i]+align_peaks_x[h*2][i+1]+align_peaks_x[h*2+2][i]+align_peaks_x[h*2+2][i+1])/4 for i in range(len(align_peaks_x[h*2])-1) ] for h in range(int(col/2-1+col%2)) ])    # x座標の重心の計算
            center_y = np.array([ [ (align_peaks_y[h*2][i]+align_peaks_y[h*2][i+1]+align_peaks_y[h*2+2][i]+align_peaks_y[h*2+2][i+1])/4 for i in range(len(align_peaks_x[h*2])-1) ] for h in range(int(col/2-1+col%2)) ])    # y座標の重心の計算
            distance_x = [ align_peaks_x[i*2+1] for i in range(int(col/2-1+col%2)) ] - center_x    # x座標に関して、Srカラムの重心とTiカラムの差分を計算 
            distance_y = [ align_peaks_y[i*2+1] for i in range(int(col/2-1+col%2)) ] - center_y    # y座標に関して、Srカラムの重心とTiカラムの差分を計算
    if bottom_row == "Ti":
        if left_col == "Ti":
            center_x = np.array([ [ (align_peaks_x[h*2+1][i]+align_peaks_x[h*2+1][i+1]+align_peaks_x[h*2+3][i]+align_peaks_x[h*2+3][i+1])/4 for i in range(len(align_peaks_x[h*2+1])-1) ] for h in range(int(col/2-1)) ])    # x座標の重心の計算
            center_y = np.array([ [ (align_peaks_y[h*2+1][i]+align_peaks_y[h*2+1][i+1]+align_peaks_y[h*2+3][i]+align_peaks_y[h*2+3][i+1])/4 for i in range(len(align_peaks_x[h*2+1])-1) ] for h in range(int(col/2-1)) ])    # y座標の重心の計算
            distance_x = [ align_peaks_x[i*2][:-1] for i in range(1, int(col/2)) ] - center_x         # x座標に関して、Srカラムの重心とTiカラムの差分を計算
            distance_y = [ align_peaks_y[i*2][:-1] for i in range(1, int(col/2)) ] - center_y         # y座標に関して、Srカラムの重心とTiカラムの差分を計算
        elif left_col == "Sr":
            center_x = np.array([ [ (align_peaks_x[h*2+1][i]+align_peaks_x[h*2+1][i+1]+align_peaks_x[h*2+3][i]+align_peaks_x[h*2+3][i+1])/4 for i in range(len(align_peaks_x[h*2+1])-1) ] for h in range(int(col/2-1)) ])    # x座標の重心の計算
            center_y = np.array([ [ (align_peaks_y[h*2+1][i]+align_peaks_y[h*2+1][i+1]+align_peaks_y[h*2+3][i]+align_peaks_y[h*2+3][i+1])/4 for i in range(len(align_peaks_x[h*2+1])-1) ] for h in range(int(col/2-1)) ])    # y座標の重心の計算
            distance_x = [ align_peaks_x[i*2] for i in range(1, int(col/2)) ] - center_x         # x座標に関して、Srカラムの重心とTiカラムの差分を計算
            distance_y = [ align_peaks_y[i*2] for i in range(1, int(col/2)) ] - center_y         # y座標に関して、Srカラムの重心とTiカラムの差分を計算       

    flat_center_x = [ e for row in center_x for e in row ]        # 1次元ベクトル化
    flat_center_y = [ e for row in center_y for e in row ]        # 1次元ベクトル化
    flat_distance_x = [ e for row in distance_x for e in row ]    # 1次元ベクトル化
    flat_distance_y = [ e for row in distance_y for e in row ]    # 1次元ベクトル化
    distances = [ ((x**2)+(y**2))**(1/2) for (x,y) in zip(flat_distance_x, flat_distance_y) ]

    return distances, flat_distance_x, flat_distance_y, flat_align_peaks_x, flat_align_peaks_y, flat_center_x, flat_center_y

# Calculate shift probabilities in 8 directions
def shift_probability(pred_flat_distance_x, pred_flat_distance_y):
    ranges = [(0, 22.5), (22.5, 67.5), (67.5, 112.5), (112.5, 157.5), (157.5, 202.5), (202.5, 247.5), (247.5, 292.5), (292.5, 337.5), (337.5, 360)]

    probabilities = []

    for i in range(len(pred_flat_distance_x[0])):
        x = [row[i] for row in pred_flat_distance_x]
        y = [row[i] for row in pred_flat_distance_y]
        angle = np.arctan2(y, x) * (180 / np.pi)

        angle = np.where(angle < 0, angle + 360, angle)

        probabilities_i = []
        total_count = 0
        for r in ranges:
            count = np.sum((r[0] <= angle) & (angle < r[1]))
            probabilities_i.append(count)
            total_count += count

        # Combine the first and last ranges to cover 337.5° to 22.5°
        probabilities_i[0] += probabilities_i[-1]
        probabilities_i.pop(-1)

        probabilities_i = [p / total_count for p in probabilities_i]
        probabilities.append(probabilities_i)

    return probabilities