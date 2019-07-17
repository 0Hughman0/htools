from scipy import interpolate


def find_FWHM(df, ylabel, xlabel=None, min_corr=False):
    """
    Find the FWHM ylabel some data
    
    df : DataFrame
        DataFrame containing the data
    ylabel : str
        Column containing ylabel values
    xlabel : str
        Column containing xlabel values, defualt is None in which case uses df.index
    min_corr : bool
        Correct for minima not being at zero i.e. shift all y values down by `y.min()`

    Returns
    =======
    FWHM : float
        Width of peak at half maximum AKA FWHM 
    (x_left, x_right) : tuple
        xlabel coords of boundary of peak
    HM : float
        Half maximum value   
    """
    y = df[ylabel]
    x = df[xlabel] if xlabel else df.index
    MAX = y.max()
    MIN = y.min()
    HM = MIN + ((MAX - MIN) / 2) if min_corr else MAX / 2

    # Fixes problem when uses indexes - things are confusing with indexes that contain 1 stepped integers!
    
    isplit = df.index.get_loc(y.idxmax())
    #else:
        # isplit = df.loc[y.idxmax(), xlabel]

    left_slopex, left_slopey = x.values[:isplit], y.values[:isplit]
    right_slopex, right_slopey = x.values[isplit:], y.values[isplit:]

    left_f = interpolate.interp1d(left_slopey, left_slopex)
    right_f = interpolate.interp1d(right_slopey, right_slopex)

    x_left = float(left_f(HM))
    x_right = float(right_f(HM))

    FW = x_right - x_left
    
    return FW, (x_left, x_right), HM
