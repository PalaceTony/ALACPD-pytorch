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


def SetArguments(
    data,
    filename,
    window=24 * 7,
    horizon=12,
    CNNFilters=100,
    CNNKernel=6,
    GRUUnits=100,
    SkipGRUUnits=5,
    skip=24,
    dropout=0.2,
    normalize=2,
    highway=24,
    lr=0.001,
    batchsize=128,
    epochs=100,
    initializer="glorot_uniform",
    loss="mean_squared_error",
    optimizer="Adam",
    trainpercent=0.6,
    validpercent=0.2,
    save=None,
    load=None,
    tensorboard=None,
    predict=None,
    series_to_plot="0",
    autocorrelation=None,
    save_plot=None,
    no_train=False,
    no_validation=False,
    test=False,
    no_saveresults=False,
    savehistory=False,
    plot=False,
    no_log=False,
    debuglevel=20,
    logfilename="log/cpdnet",
):
    args = {}
    args["data"] = data
    args["window"] = window
    args["horizon"] = horizon
    args["CNNFilters"] = CNNFilters
    args["CNNKernel"] = CNNKernel
    args["GRUUnits"] = GRUUnits
    args["SkipGRUUnits"] = SkipGRUUnits
    args["skip"] = skip
    args["dropout"] = dropout
    args["normalize"] = normalize
    args["highway"] = highway
    args["lr"] = lr
    args["batchsize"] = batchsize
    args["epochs"] = epochs
    args["initializer"] = initializer
    args["loss"] = loss
    args["optimizer"] = optimizer
    args["trainpercent"] = trainpercent
    args["validpercent"] = validpercent
    args["save"] = save
    args["load"] = load
    args["tensorboard"] = tensorboard
    args["predict"] = predict
    args["series_to_plot"] = series_to_plot
    args["autocorrelation"] = autocorrelation
    args["save_plot"] = save_plot
    args["no_train"] = no_train
    args["no_validation"] = no_validation
    args["test"] = test
    args["no_saveresults"] = no_saveresults
    args["savehistory"] = savehistory
    args["plot"] = plot
    args["no_log"] = no_log
    args["debuglevel"] = debuglevel
    args["logfilename"] = filename + logfilename
    return args
