import numpy as np

def createFirstGen(X0, N):
    try:
        p = X0.shape[1]
    except:
        p = len(X0)
    return np.random.normal(X0, size=(N, p))

def createNextGen(X1, g, N):
    M = X1.shape[0]
    X = np.zeros((N, X1.shape[1]))
    
    # keep the old best ones
    X[0:M, :] = X1
    
    # shuffle the best ones
    ridx = np.random.choice(range(M), size=N-M, replace=True)
    X1 = X1[ridx, :]
    
    # normal sample with a slow decay of sd with generation
    X[M:N, :] = np.random.normal(X1, scale=np.exp(-g/25))
    return X

def Minimize(fn, X0, M = 10, N = 100, G = 100, trace=False):
    assert callable(fn)
    assert isinstance(X0, np.ndarray), \
            f"Please provide X0 as an ndarray, not {type(X0)}"
    assert M <= N
    
    if np.any(np.abs(X0) > 3):
        print(f"WARNING: X0 contains extreme values,"
              "please make sure you are minimizing with standard scaled inputs.")

    for g in range(G):
        if g > 0:
            X = createNextGen(X1, g, N)
        else:
            X = createFirstGen(X0, N)
        
        y = fn(X)
        assert isinstance(y, np.ndarray), f"fn() must return an ndarray, not {type(y)}"
        idx = np.argsort(y, axis=None)
        
        # M best
        y = y[idx][:M]
        X1 = X[idx][:M]
        
        if trace:
            print(f"Gen {g:02d} best values {y}")
        
    return X1[0, :]
