def only_single_unit(df):
    df_filtered = df[df['calculatedfinishedsquarefeet'] < 5000]
    df_filtered = df_filtered[df_filtered['calculatedfinishedsquarefeet'] >=1000]
    df_filtered = df_filtered[df_filtered['bedroomcnt'] >=3]
    df_filtered = df_filtered[df_filtered['bedroomcnt'] <= 10]
    df_filtered = df_filtered[df_filtered['bathroomcnt'] >2]
    df_filtered = df_filtered[df_filtered['bathroomcnt'] <7]
    return df_filtered

def train_validate_test(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns train, validate, test sets and also another 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

        
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test

def handle_outliers(df, col):
    q1 = df[col].quantile(.25)
    q3 = df[col].quantile(.75)
    iqr = q3-q1 #Interquartile range
    lower_bound  = q1-1.5*iqr
    upper_bound = q3+1.5*iqr
    if lower_bound < 0:
        lower_bound = 0
    if upper_bound > df[col].max():
        upper_bound = df[col].max()
    df_out = df.loc[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df_out

def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.75):
    threshold = int(round(prop_required_columns * len(df.index),0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns),0))
    df = df.dropna(axis=0, thresh=threshold)
    return df