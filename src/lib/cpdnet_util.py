class CPDNetInit(object):
    def __init__(
        self,
        args,
        skip,
    ):
        self.data = "data/" + args.dataset_name + ".txt"
        self.window = args["windows"]
        self.horizon = args["horizon"]
        self.CNNFilters = args["CNNFilters"]
        self.CNNKernel = args["CNNKernel"]
        self.GRUUnits = args["GRUUnits"]
        self.SkipGRUUnits = args["SkipGRUUnits"]
        self.skip = skip
        self.dropout = args["dropout"]
        self.normalise = args["normalize"]
        self.highway = args["highway"]
        self.batchsize = args["batchsize"]
        self.epochs = args["epochs"]
        self.initialiser = args["initializer"]
        self.trainpercent = args["train_percent"]
        self.validpercent = args["valid_percent"]
        self.train = True
        self.lr = args["lr"]
        self.optimiser = "Adam"
