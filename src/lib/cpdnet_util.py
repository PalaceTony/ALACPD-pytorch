class CPDNetInit(object):
    def __init__(self, args, args_is_dictionary=False):
        if args_is_dictionary is True:
            self.data = args["data"]
            self.window = args["window"]
            self.horizon = args["horizon"]
            self.CNNFilters = args["CNNFilters"]
            self.CNNKernel = args["CNNKernel"]
            self.GRUUnits = args["GRUUnits"]
            self.SkipGRUUnits = args["SkipGRUUnits"]
            self.skip = args["skip"]
            self.dropout = args["dropout"]
            self.normalise = args["normalize"]
            self.highway = args["highway"]
            self.batchsize = args["batchsize"]
            self.epochs = args["epochs"]
            self.initialiser = args["initializer"]
            self.trainpercent = args["trainpercent"]
            self.validpercent = args["validpercent"]
            self.highway = args["highway"]
            self.train = not args["no_train"]
            self.validate = not args["no_validation"]
            self.save = args["save"]
            self.saveresults = not args["no_saveresults"]
            self.savehistory = args["savehistory"]
            self.load = args["load"]
            self.loss = args["loss"]
            self.lr = args["lr"]
            self.optimiser = args["optimizer"]
            self.evaltest = args["test"]
            self.tensorboard = args["tensorboard"]
            self.plot = args["plot"]
            self.predict = args["predict"]
            self.series_to_plot = args["series_to_plot"]
            self.autocorrelation = args["autocorrelation"]
            self.save_plot = args["save_plot"]
            self.log = not args["no_log"]
            self.debuglevel = args["debuglevel"]
            self.logfilename = args["logfilename"]
        else:
            self.data = args.data
            self.window = args.window
            self.horizon = args.horizon
            self.CNNFilters = args.CNNFilters
            self.CNNKernel = args.CNNKernel
            self.GRUUnits = args.GRUUnits
            self.SkipGRUUnits = args.SkipGRUUnits
            self.skip = args.skip
            self.dropout = args.dropout
            self.normalise = args.normalize
            self.highway = args.highway
            self.batchsize = args.batchsize
            self.epochs = args.epochs
            self.initialiser = args.initializer
            self.trainpercent = args.trainpercent
            self.validpercent = args.validpercent
            self.highway = args.highway
            self.train = not args.no_train
            self.validate = not args.no_validation
            self.save = args.save
            self.saveresults = not args.no_saveresults
            self.savehistory = args.savehistory
            self.load = args.load
            self.loss = args.loss
            self.lr = args.lr
            self.optimiser = args.optimizer
            self.evaltest = args.test
            self.tensorboard = args.tensorboard
            self.plot = args.plot
            self.predict = args.predict
            self.series_to_plot = args.series_to_plot
            self.autocorrelation = args.autocorrelation
            self.save_plot = args.save_plot
            self.log = not args.no_log
            self.debuglevel = args.debuglevel
            self.logfilename = args.logfilename

    def dump(self):
        print("Data: %s", self.data)
        print("Window: %d", self.window)
        print("Horizon: %d", self.horizon)
        print("CNN Filters: %d", self.CNNFilters)
        print("CNN Kernel: %d", self.CNNKernel)
        print("GRU Units: %d", self.GRUUnits)
        print("Skip GRU Units: %d", self.SkipGRUUnits)
        print("Skip: %d", self.skip)
        print("Dropout: %f", self.dropout)
        print("Normalise: %d", self.normalise)
        print("Highway: %d", self.highway)
        print("Batch size: %d", self.batchsize)
        print("Epochs: %d", self.epochs)
        print("Learning rate: %s", str(self.lr))
        print("Initialiser: %s", self.initialiser)
        print("Optimiser: %s", self.optimiser)
        print("Loss function to use: %s", self.loss)
        print("Fraction of data to be used for training: %.2f", self.trainpercent)
        print("Fraction of data to be used for validation: %.2f", self.validpercent)
        print("Train model: %s", self.train)
        print("Validate model: %s", self.validate)
        print("Test model: %s", self.evaltest)
        print("Save model location: %s", self.save)
        print("Save Results: %s", self.saveresults)
        print("Save History: %s", self.savehistory)
        print("Load Model from: %s", self.load)
        print("TensorBoard: %s", self.tensorboard)
        print("Plot: %s", self.plot)
        print("Predict: %s", self.predict)
        print("Series to plot: %s", self.series_to_plot)
        print("Save plot: %s", self.save_plot)
        print("Create log: %s", self.log)
        print("Debug level: %d", self.debuglevel)
        print("Logfile: %s", self.logfilename)


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
