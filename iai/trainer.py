from iai.imports import *
from iai.sequenceBatchGenerator import *
from iai.helpers import *
from iai.simulator import *
from iai.networks import *

def runModels_adaptive(ModelFuncPointer,
            ModelName,
            TrainDir,
            ValiDir,
            TestDir,
            NetworkDir,
            ProjectDir,
            TrainGenerator,
            ValidationGenerator,
            TestGenerator,
            TrainParams=None,
            ValiParams=None,
            TestParams=None,
            resultsFile=None,
            numEpochs=10,
            epochSteps=100,
            validationSteps=1,
            initModel=None,
            initWeights=None,
            network=None,
            nCPU = 1,
            gpuID = 0,
            rep=None,
            admixture=None,
            testGrid=0,
            gridParams=None):

    if gridParams:
        gridPars = gridParams.split(",")
    else:
        gridPars = []


    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuID)


    ## The following code block appears necessary for running with tf2 and cudnn
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import Session
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    Session(config=config)
    ###


    if(resultsFile == None):
        resultsFilename = os.path.basename(trainFile)[:-4] + ".p"
        resultsFile = os.path.join("./results/",resultsFilename)

    # Redefine modelSave and weightsSave
    resultsFile = resultsFile.replace(".p","_adapt_1.p")
    weightsSave = network[1].replace(".h5","_adapt_1.h5")
    modelSave = network[0]

    # If TestGenerator is called after model.fit the random shuffling is not the same, even with same seed
    print("\nReading test set...")
    x_test,y_test = TestGenerator.__getitem__(0)

    ct = 1
    last_acc = 0.0
    acc_diff = 1.0
    while acc_diff >= 0.001:
        print("Adaptive training iteration %s..."%(ct))
        if ct > 1:
            ## Resimulate using same params as those if tail of accuracy dist
            ## Identify test examples with lowest accuracy
            #resim_ids = []
            #deviation = []
            #accuracy = []
            #for i in range(y_test.shape[0]):
            #    if y_test[i][0] == 1.0:
            #        D = 1.0 - y_pred[i][0]
            #        deviation.append([D,i])
            #        accuracy.append(1.0-D)
            #    else:
            #        D = 1.0 - y_pred[i][1]
            #        deviation.append([D,i])
            #        accuracy.append(1.0-D)
            #deviation = sorted(deviation)[math.ceil(y_test.shape[0]/10.0)*-1:]
            #for d in deviation:
            #    resim_ids.append(d[1])
            #resim_ids = np.array(resim_ids)
            #mask = np.zeros(y_test.shape[0], dtype=bool)
            #mask[resim_ids] = True

            ## Create directories for new training sims
            #newTrainDir = TrainParams["treesDirectory"] + "_adapt"
            #newValiDir = ValiParams["treesDirectory"] + "_adapt"
            #for d in [newTrainDir, newValiDir]:
            #    if os.path.exists(d):
            #        shutil.rmtree(d)
            #    os.mkdir(d)

            ## Resimulate using new parameters
            #dg_params = pickle.load(open(os.path.join(NetworkDir, "simPars.p"), "rb"))
            #test_params = pickle.load(open(os.path.join(TestDir, "info.p"), "rb"))

            #dg_train = Simulator(**dg_params)
            #dg_vali = Simulator(**dg_params)
            #dg_train.simulateAndProduceTrees(numReps=np.sum(mask)*100,direc=newTrainDir,simulator="msprime",nProc=nCPU,test_params=test_params,mask=mask)
            #dg_vali.simulateAndProduceTrees(numReps=np.sum(mask)*5,direc=newValiDir,simulator="msprime",nProc=nCPU,test_params=test_params,mask=mask)

            ## Redefine the batch generators
            #TrainGenerator = SequenceBatchGenerator(**TrainParams)
            #ValidationGenerator = SequenceBatchGenerator(**ValiParams)

            ## Prep for loading weights from previous training iteration
            #resultsFile = resultsFile.replace("_adapt_%s.p"%(ct-1),"_adapt_%s.p" %(ct))
            #initModel = modelSave
            #initWeights = weightsSave
            #weightsSave = weightsSave.replace("_adapt_%s.h5"%(ct-1),"_adapt_%s.h5"%(ct))


            ## Resimulate, with sampling density of parameters is inversly proportional to test accuracy
            train_params = pickle.load(open(os.path.join(TrainDir, "info.p"), "rb"))
            vali_params = pickle.load(open(os.path.join(ValiDir, "info.p"), "rb"))
            test_params = pickle.load(open(os.path.join(TestDir, "info.p"), "rb"))
            x_unique = np.unique(test_params[gridPars[0]])
            y_unique = np.unique(test_params[gridPars[1]])
            X, Y = np.meshgrid(x_unique, y_unique)

            # Identify test examples with lowest accuracy
            # there has to be a better way to do this using np.where
            binary_acc = []
            print("Accuracy calculation...")
            for i in range(y_test.shape[0]):
                progress_bar((i+1)/y_test.shape[0])
                if y_test[i][0] == 1.0:
                    if y_pred[i][0] >= 0.5:
                        binary_acc.append(1)
                    else:
                        binary_acc.append(0)
                else:
                    if y_pred[i][1] >= 0.5:
                        binary_acc.append(1)
                    else:
                        binary_acc.append(0)
            binary_acc = np.array(binary_acc)

            print("\n")
            nTrain = int(train_params["numReps"] * 1.0) # what is the size on the new training set relative to the original generated by iai-simulate
            nVali = int(vali_params["numReps"] * 1.0)
            nTest = int(test_params["numReps"]/(testGrid**2))

            z = binary_acc
            z = np.reshape(z, (x_unique.shape[0],
                y_unique.shape[0],
                int(np.divide(z.shape[0],
                    x_unique.shape[0] * y_unique.shape[0]))))

            #print(np.sum(z,axis=2)+1)
            z = np.divide(1,np.sum(z,axis=2)+1) # add 1 (so as to avoid divide by zero) and take inverse
            #print(z)
            c = 1 # increasing the constant, c, makes the weighting more extreme
            z_norm = z**c / np.sum(z**c) # normalize
            #print(z_norm)
            #z_norm = z / np.linalg.norm(z) #Frobenius 2-norm
            #z_norm /= z_norm.shape[0]
            zTrain = np.rint(z_norm * nTrain)
            zVali = np.rint(z_norm * nVali)
            #print(zTrain)
            #print(np.sum(zTrain))
            #sys.exit()

            # Create directories for new training sims
            newTrainDir = TrainParams["treesDirectory"] + "_adapt"
            newValiDir = ValiParams["treesDirectory"] + "_adapt"
            newTestDir = TestParams["treesDirectory"] + "_adapt"
            for d in [newTrainDir, newValiDir, newTestDir]:
                if os.path.exists(d):
                    shutil.rmtree(d)
                os.mkdir(d)

            # Use new parameters
            dg_params = pickle.load(open(os.path.join(NetworkDir, "simPars.p"), "rb"))
            dg_params["seed"] = ct

            dg_train = Simulator(**dg_params)
            dg_vali = Simulator(**dg_params)
            # set testGrid for the new test set
            dg_params["testGrid"] = testGrid
            dg_params["gridParams"] = gridParams
            dg_test = Simulator(**dg_params)

            # Simulate
            print("Simulate train set:")
            dg_train.simulateAndProduceTrees(numReps=int(np.sum(zTrain)),direc=newTrainDir,simulator="msprime",nProc=nCPU,X=X,Y=Y,Z=zTrain,gridPars=gridPars)
            print("Simulate vali set:")
            dg_vali.simulateAndProduceTrees(numReps=int(np.sum(zVali)),direc=newValiDir,simulator="msprime",nProc=nCPU,X=X,Y=Y,Z=zVali,gridPars=gridPars)
            print("Simulate test set:")
            dg_test.simulateAndProduceTrees(numReps=nTest,direc=newTestDir,simulator="msprime",nProc=nCPU)

            # Redefine the batch generators
            TrainGenerator = SequenceBatchGenerator(**TrainParams)

            ValiParams['batchSize'] = 64
            ValidationGenerator = SequenceBatchGenerator(**ValiParams)

            TestParams['batchSize'] = nTest*(testGrid**2)
            TestParams['shuffleExamples'] = False
            TestGenerator = SequenceBatchGenerator(**TestParams)

            # Prep for loading weights from previous training iteration
            resultsFile = resultsFile.replace("_adapt_%s.p"%(ct-1),"_adapt_%s.p" %(ct))
            initModel = modelSave
            initWeights = weightsSave
            weightsSave = weightsSave.replace("_adapt_%s.h5"%(ct-1),"_adapt_%s.h5"%(ct))

            # Regenerate new test set
            x_test,y_test = TestGenerator.__getitem__(0)



        # Call the training generator
        x,y = TrainGenerator.__getitem__(0)

        ## define model
        model = ModelFuncPointer(x,y)
        # Early stopping and saving the best weights
        if ct > 1:
            patience = 50
        else:
            patience = 50
        callbacks_list = [
                EarlyStopping(
                    monitor='val_loss',
                    verbose=1,
                    min_delta=0.01,
                    patience=patience),
                ModelCheckpoint(
                    filepath=weightsSave,
                    monitor='val_loss',
                    save_best_only=True)
                ]
        if initWeights:
            print("Loading model/weights from path!")
            assert initModel != None
            jsonFILE = open(initModel,"r")
            loadedModel = jsonFILE.read()
            jsonFILE.close()
            model=model_from_json(loadedModel)
            model.load_weights(initWeights)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        ### include conditional for testing
        #if ct > 1: ## include conditional for testing
        history = model.fit(TrainGenerator,
            steps_per_epoch=epochSteps,
            epochs=numEpochs,
            validation_data=ValidationGenerator,
            use_multiprocessing=False,
            callbacks=callbacks_list,
            verbose=2)

        # Write the network
        if(network != None):
            ##serialize model to JSON
            model_json = model.to_json()
            with open(modelSave, "w") as json_file:
                json_file.write(model_json)

        # Load json and create model
        if(network != None):
            jsonFILE = open(modelSave,"r")
            loadedModel = jsonFILE.read()
            jsonFILE.close()
            model=model_from_json(loadedModel)
            model.load_weights(weightsSave)
        else:
            print("Error: model and weights not loaded")
            sys.exit(1)

        # Metrics to track the different accuracies.
        test_acc = tf.metrics.CategoricalAccuracy()

        # Predict on clean test examples
        print("Predicting...")
        y_pred = model.predict(x_test)
        test_acc(y_test, y_pred)
        new_acc = float(test_acc.result())
        print('\nAdaptive iteration %s: test acc: %s' %(ct, new_acc))
        print("Results written to: ",resultsFile)
        history.history['loss'] = np.array(history.history['loss'])
        history.history['val_loss'] = np.array(history.history['val_loss'])
        history.history['predictions'] = np.array(y_pred)
        history.history['Y_test'] = np.array(y_test)
        history.history['name'] = ModelName
        pickle.dump(history.history, open(resultsFile, "wb" ))

        # Evaluate improvement in accuracy
        acc_diff = new_acc - last_acc
        last_acc = new_acc
        if ct > 1:
            print("\nAccuracy improvement relative to last iteration:",acc_diff)

        # Plot training results
        plotResultsSoftmax2Heatmap(resultsFile=resultsFile,saveas=resultsFile.replace(".p",".pdf"),admixture=admixture)
        ct+=1
        #if ct > 2:
        #    break


def predict_adaptive(ModelFuncPointer,
            ModelName,
            NetworkDir,
            ProjectDir,
            TrainGenerator,
            ValidationGenerator,
            TestGenerator,
            test_info=None,
            resultsFile=None,
            numEpochs=10,
            epochSteps=100,
            validationSteps=1,
            init=None,
            network=None,
            nCPU = 1,
            gpuID = 0,
            paramsID = None,
            admixture=None):

    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuID)

    ## The following code block appears necessary for running with tf2 and cudnn
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import Session
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    Session(config=config)
    ###

    # Redefine modelSave and weightsSave
    resultsFile = resultsFile.replace(".p","_adapt_1.p")
    weightsSave = network[1].replace(".h5","_adapt_1.h5")
    modelSave = network[0]

    ########### Prediction on adaptive iteration 1 #############
    # Load json and create model
    if(network != None):
        jsonFILE = open(modelSave,"r")
        loadedModel = jsonFILE.read()
        jsonFILE.close()
        model=model_from_json(loadedModel)
        model.load_weights(weightsSave)
    else:
        print("Error: model and weights not loaded")
        sys.exit(1)

    # Metrics to track the different accuracies.
    test_acc_clean = tf.metrics.CategoricalAccuracy()
    test_acc_adapt = tf.metrics.CategoricalAccuracy()

    # Read all clean test data into memory
    x_test, y_test = TestGenerator.__getitem__(0)
    predictions = model.predict(x_test)

    #replace predictions and Y_test in results file

    history= pickle.load(open(resultsFile, "rb"))
    tmp = []
    for gr in test_info["gr"]:
        if gr > 0.0:
            tmp.append([0.0,1.0])
        else:
            tmp.append([1.0,0.0])
    history["Y_test"] = np.array(tmp)
    history['predictions'] = np.array(predictions)

    #rewrite result file
    newResultsFile = resultsFile.replace(".p","_params%s.p"%(paramsID))
    print("new results written to: ",newResultsFile)
    pickle.dump(history, open(newResultsFile, "wb"))
    test_acc_clean(y_test, predictions)

    # Determine number of adaptive iterations with improved accuracy
    nthIter = 0
    for f in glob.glob(os.path.join(NetworkDir,"testResults_adapt_*.pdf")):
        nthIter+=1
    nthIter-=1

    ########### Prediction on final adpative iteration #############
    # Redefine modelSave and weightsSave
    resultsFile = resultsFile.replace("_adapt_1.p","_adapt_%s.p"%(nthIter))
    weightsSave = network[1].replace(".h5","_adapt_%s.h5"%(nthIter))

    # Load json and create model
    if(network != None):
        jsonFILE = open(modelSave,"r")
        loadedModel_adapt = jsonFILE.read()
        jsonFILE.close()
        model_adapt=model_from_json(loadedModel_adapt)
        model_adapt.load_weights(weightsSave)
    else:
        print("Error: model_adapt and weights_adapt not loaded")
        sys.exit(1)

    predictions_adapt = model_adapt.predict(x_test)

    #replace predictions and T_test in results file
    history_adapt = pickle.load(open(resultsFile, "rb"))
    tmp = []
    for gr in test_info["gr"]:
        if gr > 0.0:
            tmp.append([0.0,1.0])
        else:
            tmp.append([1.0,0.0])
    history_adapt["Y_test"] = np.array(tmp)
    history_adapt['predictions'] = np.array(predictions_adapt)
    test_acc_adapt(y_test, predictions_adapt)

    # rewrite new results file
    newResultsFile = resultsFile.replace(".p","_params%s.p"%(paramsID))
    print("new results written to: ", newResultsFile)
    pickle.dump(history_adapt, open(newResultsFile, "wb"))

    # Plot results
    plotResultsSoftmax2HeatmapMis(resultsFile=newResultsFile.replace("_adapt_%s_params%s.p"%(nthIter,paramsID),"_adapt_1_params%s.p"%(paramsID)),
            resultsFile2=newResultsFile,
            saveas=newResultsFile.replace(".p",".pdf"),
            admixture=admixture)

    ######### write log ###########
    outLog = resultsFile.replace("_adapt_%s.p"%(nthIter),"_log_params%s.txt"%(paramsID))
    with open(outLog, "w") as fOUT:
        fOUT.write("Before adaptive training\n")
        fOUT.write("===========================\n")
        fOUT.write('test acc on test_params2 examples (%): {:.3f}\n'.format(test_acc_clean.result() * 100))
        fOUT.write("\nAfter adaptive training (%s iterations of improvement)\n"%(nthIter-1))
        fOUT.write("===========================\n")
        fOUT.write('test acc on test_params2 examples (%): {:.3f}\n'.format(test_acc_adapt.result() * 100))

    return None
