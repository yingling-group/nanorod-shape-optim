grid = [
    # obs or imputed
    [ObservedDF(), ImputedDF()],
    # Feature aggregation
    [Aggregate()],
    # Initial features
    [NonCollinearFeatures(), ValidFeatures()],
    # Scale X
    [ScaleX(), None],
    # Augmentation method
    [Augment('random'), Augment('smote'), Augment('quality')],
    # Feature selection
    [RFCFeatureSelect(), SFSFeatureSelec()],
    # Algorithm choice
    [KNN(), SVC(), RFC(), XGB(), GPR()],
    # Hyperparam search
    [HyperParam()],
    # Fit the model with hyper params and features
    Fit()
]

class Payload:
    def __init__(self):
        self.Tr = None 
        self.Ts = None 
        self.xCols = None 
        self.yCol = None
        self.xsclr = None 
        self.ysclr = None

    def set(self, **kwargs):
        for kw in kwargs:
            v = kwargs[kw]
            if kw == 'Tr':
                assert isinstance(v, pd.DataFrame)
                self.Tr = v
            elif kw == 'Ts':
                assert isinstance(v, pd.DataFrame)
                self.Ts = v
            elif kw.lower() == 'xcols':
                assert isinstance(v, pd.Series) or isinstance(v, list)
                self.xCols = v
            elif kw.lower() == 'ycol':
                assert isinstance(v, str)
                self.yCol = v
            else:
                raise NameError(kw)


items = itertools.product(grid)
