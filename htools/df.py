from scipy import interpolate


def find_FWHM(df, ylabel, xlabel=None):
    """
    Find the FWHM ylabel some data
    
    df : DataFrame
        DataFrame containing the data
    ylabel : str
        Column containing ylabel values
    xlabel : str
        Column containing xlabel values, defualt is None in which case uses df.index

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
    HM =  MIN + ((MAX - MIN) / 2)
    DX = HM / 2
    isplit = y.idxmax()
    left_slopex, left_slopey = x[:isplit], y[:isplit]
    right_slopex, right_slopey = x[isplit:], y[isplit:]

    left_f = interpolate.interp1d(left_slopey, left_slopex)
    right_f = interpolate.interp1d(right_slopey, right_slopex)

    x_left = float(left_f(HM))
    x_right = float(right_f(HM))

    FW = x_right - x_left
    
    return FW, (x_left, x_right), HM
