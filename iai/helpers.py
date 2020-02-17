'''
Authors: Jared Galloway, Jeff Adrion
'''

from iai.imports import *

#-------------------------------------------------------------------------------------------

def assign_task(mpID, task_q, nProcs):
    c,i,nth_job=0,0,1
    while (i+1)*nProcs <= len(mpID):
        i+=1
    nP1=nProcs-(len(mpID)%nProcs)
    for j in range(nP1):
        task_q.put((mpID[c:c+i], nth_job))
        nth_job += 1
        c=c+i
    for j in range(nProcs-nP1):
        task_q.put((mpID[c:c+i+1], nth_job))
        nth_job += 1
        c=c+i+1

#-------------------------------------------------------------------------------------------

def create_procs(nProcs, task_q, result_q, params, worker):
    pids = []
    for _ in range(nProcs):
        p = mp.Process(target=worker, args=(task_q, result_q, params))
        p.daemon = True
        p.start()
        pids.append(p)
    return pids

#-------------------------------------------------------------------------------------------

def get_corrected_index(L,N):
    idx,outN="",""
    dist=float("inf")
    for i in range(len(L)):
        D=abs(N-L[i])
        if D < dist:
            idx=i
            outN=L[i]
            dist=D
    return [idx,outN]

#-------------------------------------------------------------------------------------------

def get_corrected(rate,bs):
    idx=get_corrected_index(bs["Q2"],rate)
    CI95LO=bs["CI95LO"][idx[0]]
    CI95HI=bs["CI95HI"][idx[0]]
    cRATE=relu(rate+(bs["rho"][idx[0]]-idx[1]))
    ciHI=relu(cRATE+(CI95HI-idx[1]))
    ciLO=relu(cRATE+(CI95LO-idx[1]))
    return [cRATE,ciLO,ciHI]

#-------------------------------------------------------------------------------------------

def get_index(pos, winSize):
    y=snps_per_win(pos,winSize)
    st=0
    indices=[]
    for i in range(len(y)):
        indices.append([st,st+y[i]])
        st+=y[i]
    return indices

#-------------------------------------------------------------------------------------------

def snps_per_win(pos, window_size):
    bins = np.arange(1, pos.max()+window_size, window_size) #use 1-based coordinates, per VCF standard
    y,x = np.histogram(pos,bins=bins)
    return y

#-------------------------------------------------------------------------------------------

def find_win_size(winSize, pos, winSizeMx):
    snpsWin=snps_per_win(pos,winSize)
    mn,u,mx = snpsWin.min(), int(snpsWin.mean()), snpsWin.max()
    if mx > winSizeMx:
        return [-1]
    elif mx < winSizeMx:
        return [1]
    else:
        return [winSize,mn,u,mx,len(snpsWin)]

#-------------------------------------------------------------------------------------------

def force_win_size(winSize, pos):
    snpsWin=snps_per_win(pos,winSize)
    mn,u,mx = snpsWin.min(), int(snpsWin.mean()), snpsWin.max()
    return [winSize,mn,u,mx,len(snpsWin)]

#-------------------------------------------------------------------------------------------

def maskStats(wins, last_win, mask, maxLen):
    """
    return a three-element list with the first element being the total proportion of the window that is masked,
    the second element being a list of masked positions that are relative to the windown start=0 and the window end = window length,
    and the third being the last window before breaking to expidite the next loop
    """
    chrom = wins[0].split(":")[0]
    a = wins[1]
    L = wins[2]
    b = a + L
    prop = [0.0,[],0]
    try:
        for i in range(last_win, len(mask[chrom])):
            x, y = mask[chrom][i][0], mask[chrom][i][1]
            if y < a:
                continue
            if b < x:
                return prop
            else:  # i.e. [a--b] and [x--y] overlap
                if a >= x and b <= y:
                    return [1.0, [[0,maxLen]], i]
                elif a >= x and b > y:
                    win_prop = (y-a)/float(b-a)
                    prop[0] += win_prop
                    prop[1].append([0,int(win_prop * maxLen)])
                    prop[2] = i
                elif b <= y and a < x:
                    win_prop = (b-x)/float(b-a)
                    prop[0] += win_prop
                    prop[1].append([int((1-win_prop)*maxLen),maxLen])
                    prop[2] = i
                else:
                    win_prop = (y-x)/float(b-a)
                    prop[0] += win_prop
                    prop[1].append([int(((x-a)/float(b-a))*maxLen), int(((y-a)/float(b-a))*maxLen)])
                    prop[2] = i
        return prop
    except KeyError:
        return prop

#-------------------------------------------------------------------------------------------

def check_demHist(path):
    fTypeFlag = -9
    with open(path, "r") as fIN:
        for line in fIN:
            if line.startswith("mutation_per_site"):
                fTypeFlag = 1
                break
            if line.startswith("label"):
                fTypeFlag = 2
                break
            if line.startswith("time_index"):
                fTypeFlag = 3
                break
    return fTypeFlag

#-------------------------------------------------------------------------------------------

def convert_msmc_output(results_file, mutation_rate, generation_time):
   """
   This function converts the output from msmc into a csv the will be read in for
   plotting comparison.

   MSMC outputs times and rates scaled by the mutation rate per basepair per generation.
   First, scaled times are given in units of the per-generation mutation rate.
   This means that in order to convert scaled times to generations,
   divide them by the mutation rate. In humans, we used mu=1e-8 per basepair per generation.
   To convert generations into years, multiply by the generation time, for which we used 10 years.

   To get population sizes out of coalescence rates, first take the inverse of the coalescence rate,
   scaledPopSize = 1 / lambda00. Then divide this scaled population size by 2*mu
   """
   outfile = results_file+".csv"
   out_fp = open(outfile, "w")
   in_fp = open(results_file, "r")
   header = in_fp.readline()
   out_fp.write("label,x,y\n")
   for line in in_fp:
       result = line.split()
       time = float(result[1])
       time_generation = time / mutation_rate
       time_years = time_generation * generation_time
       lambda00 = float(result[3])
       scaled_pop_size = 1 / lambda00
       size = scaled_pop_size / (2*mutation_rate)
       out_fp.write(f"pop0,{time_years},{size}\n")
   out_fp.close
   return None

#-------------------------------------------------------------------------------------------

def convert_demHist(path, nSamps, gen, fType, mu):
    swp, PC, DE = [],[],[]
    # Convert stairwayplot to msp demographic_events
    if fType == 1:
        with open(path, "r") as fIN:
            flag=0
            lCt=0
            for line in fIN:
                if flag == 1:
                    if lCt % 2 == 0:
                        swp.append(line.split())
                    lCt+=1
                if line.startswith("mutation_per_site"):
                    flag=1
        N0 = int(float(swp[0][6]))
        for i in range(len(swp)):
            if i == 0:
                PC.append(msp.PopulationConfiguration(sample_size=nSamps, initial_size=N0))
            else:
                DE.append(msp.PopulationParametersChange(time=int(float(swp[i][5])/float(gen)), initial_size=int(float(swp[i][6])), population=0))
    ## Convert MSMC to similar format to smc++
    if fType == 3:
        convert_msmc_output(path, mu, gen)
        path+=".csv"
    ## Convert smc++ or MSMC results to msp demographic_events
    if fType == 2 or fType == 3:
        with open(path, "r") as fIN:
            fIN.readline()
            for line in fIN:
                ar=line.split(",")
                swp.append([int(float(ar[1])/gen),int(float(ar[2]))])
        N0 = swp[0][1]
        for i in range(len(swp)):
            if i == 0:
                PC.append(msp.PopulationConfiguration(sample_size=nSamps, initial_size=N0))
            else:
                DE.append(msp.PopulationParametersChange(time=swp[i][0], initial_size=swp[i][1], population=0))
    dd=msp.DemographyDebugger(population_configurations=PC,
            demographic_events=DE)
    print("Simulating under the following population size history:")
    dd.print_history()
    MspD = {"population_configurations" : PC,
        "migration_matrix" : None,
        "demographic_events" : DE}
    if MspD:
        return MspD
    else:
        print("Error in converting demographic history file.")
        sys.exit(1)

#-------------------------------------------------------------------------------------------

def relu(x):
    return max(0,x)

#-------------------------------------------------------------------------------------------

def zscoreTargets(self):
    norm = self.targetNormalization
    nTargets = copy.deepcopy(self.infoDir['y'])
    if(norm == 'zscore'):
        tar_mean = np.mean(nTargets,axis=0)
        tar_sd = np.std(nTargets,axis=0)
        nTargets -= tar_mean
        nTargets = np.divide(nTargets,tar_sd,out=np.zeros_like(nTargets),where=tar_sd!=0)

#-------------------------------------------------------------------------------------------

def load_and_predictVCF(VCFGenerator,
            resultsFile=None,
            network=None,
            minS = 50,
            gpuID = 0,
            hotspots = False):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuID)

    # load json and create model
    if(network != None):
        jsonFILE = open(network[0],"r")
        loadedModel = jsonFILE.read()
        jsonFILE.close()
        model=model_from_json(loadedModel)
        model.load_weights(network[1])
    else:
        print("Error: no pretrained network found!")
        sys.exit(1)

    x,chrom,win,info,nSNPs = VCFGenerator.__getitem__(0)
    predictions = model.predict(x)

    if hotspots:
        with open(resultsFile, "w") as fOUT:
            ct=0
            fOUT.write("%s\t%s\t%s\t%s\t%s\n" %("chrom","start","end","nSites","hotspot"))
            for i in range(len(predictions)):
                if nSNPs[i] >= minS:
                    fOUT.write("%s\t%s\t%s\t%s\t%s\n" %(chrom,ct,ct+win,nSNPs[i],predictions[i][0]))
                ct+=win
    else:
        u=np.mean(info["rho"])
        sd=np.std(info["rho"])
        last = int(os.path.basename(resultsFile).split(".")[0].split("-")[-1])
        with open(resultsFile, "w") as fOUT:
            ct=0
            fOUT.write("%s\t%s\t%s\t%s\t%s\n" %("chrom","start","end","nSites","recombRate"))
            for i in range(len(predictions)):
                if nSNPs[i] >= minS:
                    fOUT.write("%s\t%s\t%s\t%s\t%s\n" %(chrom,ct,min(ct+win,last),nSNPs[i],relu(sd*predictions[i][0]+u)))
                ct+=win

    return None

#-------------------------------------------------------------------------------------------

def runModels(ModelFuncPointer,
            ModelName,
            TrainDir,
            TrainGenerator,
            ValidationGenerator,
            TestGenerator,
            resultsFile=None,
            numEpochs=10,
            epochSteps=100,
            validationSteps=1,
            init=None,
            network=None,
            nCPU = 1,
            gpuID = 0):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuID)

    if(resultsFile == None):

        resultsFilename = os.path.basename(trainFile)[:-4] + ".p"
        resultsFile = os.path.join("./results/",resultsFilename)

    x,y = TrainGenerator.__getitem__(0)
    model = ModelFuncPointer(x,y)

    if(init != None):
        model.load_weights(init)

    # Early stopping and saving the best weights
    callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                verbose=1,
                min_delta=0.01,
                patience=25),
            keras.callbacks.ModelCheckpoint(
                filepath=network[1],
                monitor='val_loss',
                save_best_only=True)
            ]

    history = model.fit_generator(TrainGenerator,
        steps_per_epoch= epochSteps,
        epochs=numEpochs,
        validation_data=ValidationGenerator,
        validation_steps=validationSteps,
        use_multiprocessing=True,
        callbacks=callbacks_list,
        max_queue_size=nCPU,
        workers=nCPU,
        )

    # Write the network
    if(network != None):
        ##serialize model to JSON
        model_json = model.to_json()
        with open(network[0], "w") as json_file:
            json_file.write(model_json)

    # Load json and create model
    if(network != None):
        jsonFILE = open(network[0],"r")
        loadedModel = jsonFILE.read()
        jsonFILE.close()
        model=model_from_json(loadedModel)
        model.load_weights(network[1])
    else:
        print("Error: model and weights not loaded")
        sys.exit(1)

    x,y = TestGenerator.__getitem__(0)
    predictions = model.predict(x)

    history.history['loss'] = np.array(history.history['loss'])
    history.history['val_loss'] = np.array(history.history['val_loss'])
    history.history['predictions'] = np.array(predictions)
    history.history['Y_test'] = np.array(y)
    history.history['name'] = ModelName

    print("results written to: ",resultsFile)
    pickle.dump(history.history, open( resultsFile, "wb" ))

    return None

#-------------------------------------------------------------------------------------------

def runModelsAdv(ModelFuncPointer,
            ModelName,
            NetworkDir,
            projectDir,
            TrainGenerator,
            ValidationGenerator,
            TestGenerator,
            resultsFile=None,
            numEpochs=10,
            epochSteps=100,
            validationSteps=1,
            init=None,
            network=None,
            nCPU = 1,
            gpuID = 0,
            learningRate=0.001
            ):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuID)

    # Force TensorFlow to use single thread to improve reproducibility
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                          inter_op_parallelism_threads=1)

    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    if(resultsFile == None):
        resultsFilename = os.path.basename(trainFile)[:-4] + ".p"
        resultsFile = os.path.join("./results/",resultsFilename)

    # Read all test data into memory
    x_train, y_train = TrainGenerator.__getitem__(0)
    x_test, y_test = TestGenerator.__getitem__(0)

    img_rows, img_cols = x_test.shape[1], x_test.shape[2]
    nb_classes = y_test.shape[1]

    ## Label smoothing (Not really sure why we'd want to do this [changes 0.0,1.0 to 0.05,0.95))
    #label_smoothing = 0.1
    #y_train -= label_smoothing * (y_train - 1. / nb_classes)

    # Define Keras model
    model = ModelFuncPointer(img_rows=img_rows, img_cols=img_cols,
                    channels=None, nb_filters=None,
                    nb_classes=nb_classes)
    print("Defined Keras model.")

    # To be able to call the model in the custom loss, we need to call it once
    model(model.input)

    # Load weights if given
    if(init != None):
        model.load_weights(init)

    # Initialize the Fast Gradient Sign Method (FGSM) attack object
    wrap = KerasModelWrapper(model)
    attack = FastGradientMethod(wrap, sess=sess)
    #attack = LBFGS(wrap, sess=sess)
    #attack = SPSA(wrap, sess=sess)
    #attack = SaliencyMapMethod(wrap, sess=sess)
    attack_params = {'eps': 1.,
                 'clip_min': 0.,
                 'clip_max': 1.}
    #attack_params = {'batch_size': 64,
    #            'clip_min': 0.,
    #            'clip_max': 1.}
    #attack_params = {'theta': 1., 'gamma': 0.1,
    #             'clip_min': 0., 'clip_max': 1.,
    #             'y_target': None}

    adv_acc_metric = get_adversarial_acc_metric(model, attack, attack_params)
    model.compile(
      optimizer=keras.optimizers.Adam(learningRate),
      loss='categorical_crossentropy',
      metrics=['accuracy', adv_acc_metric]
    )

    # Early stopping and saving the best weights
    callbacks_list = [
            keras.callbacks.EarlyStopping(
                verbose=1,
                min_delta=0.01,
                patience=25),
            keras.callbacks.ModelCheckpoint(
                filepath=network[1],
                save_best_only=True)
            ]

    history = model.fit_generator(TrainGenerator,
        steps_per_epoch= epochSteps,
        epochs=numEpochs,
        validation_data=ValidationGenerator,
        validation_steps=validationSteps,
        use_multiprocessing=True,
        callbacks=callbacks_list,
        max_queue_size=nCPU,
        workers=nCPU,
        )


    # Save genotype images for testset
    print("\nGenerating adversarial examples and writing images/predicions...")
    imageDir = os.path.join(projectDir,"test_images")
    if not os.path.exists(imageDir):
        os.makedirs(imageDir)
    for i in range(x_test.shape[0]):
        org_predFILE = os.path.join(imageDir,"examp{}_org_pred.txt".format(i))
        adv_predFILE = os.path.join(imageDir,"examp{}_adv_pred.txt".format(i))
        org_gmFILE = os.path.join(imageDir,"examp{}_org.npy".format(i))
        adv_gmFILE = os.path.join(imageDir,"examp{}_adv.npy".format(i))
        org_imageFILE = os.path.join(imageDir,"examp{}_org.png".format(i))
        adv_imageFILE = os.path.join(imageDir,"examp{}_adv.png".format(i))
        delta_imageFILE = os.path.join(imageDir,"examp{}_delta.png".format(i))
        org_example = x_test[i].reshape((1, x_test.shape[1], x_test.shape[2]))
        adv_example = attack.generate_np(org_example, **attack_params)
        pred_org = model.predict(org_example)
        pred_adv = model.predict(adv_example)
        np.savetxt(org_predFILE, pred_org)
        np.savetxt(adv_predFILE, pred_adv)
        org_image = org_example.reshape((img_rows, img_cols))
        adv_image = adv_example.reshape((img_rows, img_cols))
        delta_image = org_image - adv_image
        np.save(org_gmFILE, org_example)
        np.save(adv_gmFILE, adv_example)
        plt.imsave(org_imageFILE, org_image)
        plt.imsave(adv_imageFILE, adv_image)
        plt.imsave(delta_imageFILE, delta_image)
        progress_bar(i/float(x_test.shape[0]))

    sys.exit()
    # Evaluate the accuracy on legitimate and adversarial test examples
    report = AccuracyReport()
    _, acc, adv_acc = model.evaluate(x_test, y_test,
                                   batch_size=64,
                                   verbose=0)
    report.clean_train_clean_eval = acc
    report.clean_train_adv_eval = adv_acc
    outLog = resultsFile.replace(".p","_log.txt")
    with open(outLog, "w") as fOUT:
        fOUT.write("Before adversarial training\n")
        fOUT.write("===========================\n")
        fOUT.write("Test accuracy on legitimate examples: %0.4f\n" % acc)
        fOUT.write("Test accuracy on adversarial examples: %0.4f\n" % adv_acc)
    print('\nTest accuracy on legitimate examples: %0.4f' % acc)
    print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

    # Write the network
    if(network != None):
        ##serialize model to JSON
        model_json = model.to_json()
        with open(network[0], "w") as json_file:
            json_file.write(model_json)

    # Load json and create model
    if(network != None):
        jsonFILE = open(network[0],"r")
        loadedModel = jsonFILE.read()
        jsonFILE.close()
        model=model_from_json(loadedModel)
        model.load_weights(network[1])
    else:
        print("Error: model and weights not loaded")
        sys.exit(1)

    predictions = model.predict(x_test)

    history.history['loss'] = np.array(history.history['loss'])
    history.history['val_loss'] = np.array(history.history['val_loss'])
    history.history['predictions'] = np.array(predictions)
    history.history['Y_test'] = np.array(y_test)
    history.history['name'] = ModelName

    print("results written to: ",resultsFile)
    pickle.dump(history.history, open( resultsFile, "wb" ))



    ########## Adversarial training #############
    print("Repeating the process, using adversarial training")
    # Redefine Keras model
    model_2 = ModelFuncPointer(img_rows=img_rows, img_cols=img_cols,
                    channels=None, nb_filters=None,
                    nb_classes=nb_classes)
    model_2(model_2.input)
    wrap_2 = KerasModelWrapper(model_2)
    attack_2 = FastGradientMethod(wrap_2, sess=sess)
    #attack_2 = LBFGS(wrap_2, sess=sess)


    # Use a loss function based on legitimate and adversarial examples
    adv_loss_2 = get_adversarial_loss(model_2, attack_2, attack_params)
    adv_acc_metric_2 = get_adversarial_acc_metric(model_2, attack_2, attack_params)

    model_2.compile(
      optimizer=keras.optimizers.Adam(learningRate),
      loss=adv_loss_2,
      metrics=['accuracy', adv_acc_metric_2]
    )

    # Early stopping and saving the best weights
    callbacks_list_2 = [
            keras.callbacks.EarlyStopping(
                verbose=1,
                min_delta=0.01,
                patience=25),
            keras.callbacks.ModelCheckpoint(
                filepath=network[1].replace(".h5","_2.h5"),
                save_best_only=True)
            ]

    # Train model
    history_2 = model_2.fit_generator(TrainGenerator,
        steps_per_epoch= epochSteps,
        epochs=numEpochs,
        validation_data=ValidationGenerator,
        validation_steps=validationSteps,
        use_multiprocessing=True,
        callbacks=callbacks_list_2,
        max_queue_size=nCPU,
        workers=nCPU,
        )

    # Evaluate the accuracy on legitimate and adversarial test examples
    _, acc, adv_acc = model_2.evaluate(x_test, y_test,
                                     batch_size=64,
                                     verbose=0)

    report.adv_train_clean_eval = acc
    report.adv_train_adv_eval = adv_acc
    with open(outLog, "a") as fOUT:
        fOUT.write("\nAfter adversarial training\n")
        fOUT.write("===========================\n")
        fOUT.write("Test accuracy on legitimate examples: %0.4f\n" % acc)
        fOUT.write("Test accuracy on adversarial examples: %0.4f\n" % adv_acc)
    print('Test accuracy on legitimate examples: %0.4f' % acc)
    print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)


    # Write the network
    if(network != None):
        ##serialize model_2 to JSON
        model_json_2 = model_2.to_json()
        with open(network[0].replace(".json","_2.json"), "w") as json_file:
            json_file.write(model_json_2)

    # Load json and create model
    if(network != None):
        jsonFILE = open(network[0].replace(".json","_2.json"),"r")
        loadedModel_2 = jsonFILE.read()
        jsonFILE.close()
        model_2=model_from_json(loadedModel_2)
        model_2.load_weights(network[1].replace(".h5","_2.h5"))
    else:
        print("Error: model_2 and weights_2 not loaded")
        sys.exit(1)

    predictions_2 = model_2.predict(x_test)

    history_2.history['loss'] = np.array(history_2.history['loss'])
    history_2.history['val_loss'] = np.array(history_2.history['val_loss'])
    history_2.history['predictions'] = np.array(predictions_2)
    history_2.history['Y_test'] = np.array(y_test)
    history_2.history['name'] = ModelName+"_adversarialTrained"

    print("results written to: ",resultsFile.replace(".p","_2.p"))
    pickle.dump(history_2.history, open( resultsFile.replace(".p","_2.p"), "wb" ))

    reportFile = os.path.join(NetworkDir, "testReport.p")
    print("report written to: ",reportFile)
    pickle.dump(report, open( reportFile, "wb" ))

    return None

#-------------------------------------------------------------------------------------------

def runModelsJSMA(ModelFuncPointer,
            ModelName,
            NetworkDir,
            projectDir,
            TrainGenerator,
            ValidationGenerator,
            TestGenerator,
            resultsFile=None,
            numEpochs=10,
            epochSteps=100,
            validationSteps=1,
            init=None,
            network=None,
            nCPU = 1,
            gpuID = 0,
            learningRate=0.001
            ):


    viz_enabled = True
    nb_epochs = 10
    batch_size = 16
    learning_rate = .001
    source_samples = 10

    report = AccuracyReport()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuID)

    # Force TensorFlow to use single thread to improve reproducibility
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                          inter_op_parallelism_threads=1)

    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    if(resultsFile == None):
        resultsFilename = os.path.basename(trainFile)[:-4] + ".p"
        resultsFile = os.path.join("./results/",resultsFilename)

    # Read all test data into memory
    x_train, y_train = TrainGenerator.__getitem__(0)
    x_test, y_test = TestGenerator.__getitem__(0)

    x_train = x_train[:,:x_train.shape[2],:]
    x_test = x_test[:,:x_test.shape[2],:]


    ## Issues with only 2 targets
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
    #y_train = np.pad(y_train, ((0,0),(0,8)), "constant")
    #y_test = np.pad(y_test, ((0,0),(0,8)), "constant")

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print(x_test[0].shape, y_test[0].shape)
    #sys.exit()

    # Obtain Image Parameters
    #img_rows, img_cols = x_test.shape[1], x_test.shape[2]
    #nb_classes = y_test.shape[1]
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]
    print(img_rows, img_cols, nchannels)
    print(nb_classes)
    #sys.exit()

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
    #x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    print(x.shape,y.shape)
    #sys.exit()

    nb_filters = 64
    # Define TF model graph
    model = ModelBasicCNN('model1', nb_classes, nb_filters)
    print(model)
    #sys.exit()
    #model = ModelFuncPointer(img_rows=img_rows, img_cols=img_cols,
    #                channels=None, nb_filters=None,
    #                nb_classes=nb_classes)
    preds = model.get_logits(x)
    print(preds)
    #sys.exit()


    ## Not sure if better to use smoothing or no smoothing
    loss = CrossEntropy(model, smoothing=0.1)
    #loss = CrossEntropy(model)


    print("Defined TensorFlow model graph.")
    #sys.exit()
    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model
    train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate
    }
    sess.run(tf.global_variables_initializer())
    rng = np.random.RandomState([2017, 8, 30])
    train(sess, loss, x_train, y_train, args=train_params, rng=rng)

    #sys.exit()

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
    #assert x_test.shape[0] == test_end - test_start, x_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
    print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes - 1) +
        ' adversarial examples')

    # Keep track of success (adversarial example classified in target)
    results = np.zeros((nb_classes, source_samples), dtype='i')

    # Rate of perturbed features for each test set example and target class
    perturbations = np.zeros((nb_classes, source_samples), dtype='f')

    # Initialize our array for grid visualization
    grid_shape = (nb_classes, nb_classes, img_rows, img_cols)
    grid_viz_data = np.zeros(grid_shape, dtype='f')

    # Instantiate a SaliencyMapMethod attack object
    jsma = SaliencyMapMethod(model, sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                 'clip_min': 0., 'clip_max': 1.,
                 'y_target': None}

    figure = None
    # Loop over the samples we want to perturb into adversarial examples
    for sample_ind in xrange(0, source_samples):
        print('--------------------------------------')
        print('Attacking input %i/%i' % (sample_ind + 1, source_samples))
        sample = x_test[sample_ind:(sample_ind + 1)]
        #print(sample)
        #sys.exit()

        # We want to find an adversarial example for each possible target class
        # (i.e. all classes that differ from the label given in the dataset)
        current_class = int(np.argmax(y_test[sample_ind]))
        target_classes = other_classes(nb_classes, current_class)
        #print(current_class, target_classes)
        #sys.exit()


        ## For the grid visualization, keep original images along the diagonal
        #grid_viz_data[current_class, current_class, :, :, :] = np.reshape(
        #    sample, (img_rows, img_cols, nchannels))

        # Loop over all target classes
        for target in target_classes:
            print('Generating adv. example for target class %i' % target)

            # This call runs the Jacobian-based saliency map approach
            one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
            one_hot_target[0, target] = 1
            jsma_params['y_target'] = one_hot_target
            adv_x = jsma.generate_np(sample, **jsma_params)

            # Check if success was achieved
            res = int(model_argmax(sess, x, preds, adv_x) == target)

            # Compute number of modified features
            adv_x_reshape = adv_x.reshape(-1)
            test_in_reshape = x_test[sample_ind].reshape(-1)
            nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
            percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

            # Display the original and adversarial images side-by-side
            if viz_enabled:
                orgFILE = os.path.join("/home/jadrion/projects/i-against-i/jsma/","sample%s_target%s_org.png"%(sample_ind,target))
                advFILE = os.path.join("/home/jadrion/projects/i-against-i/jsma/","sample%s_target%s_adv.png"%(sample_ind,target))
                difFILE = os.path.join("/home/jadrion/projects/i-against-i/jsma/","sample%s_target%s_dif.png"%(sample_ind,target))
                #plt.imsave(orgFILE, np.reshape(sample, (img_rows, img_cols, nchannels)))
                #plt.imsave(advFILE, np.reshape(adv_x, (img_rows, img_cols, nchannels)))
                plt.imsave(orgFILE, np.reshape(sample, (img_rows, img_cols)))
                plt.imsave(advFILE, np.reshape(adv_x, (img_rows, img_cols)))
                plt.imsave(difFILE, np.reshape(adv_x, (img_rows, img_cols)) - np.reshape(sample, (img_rows, img_cols)))
                #figure = pair_visual(
                #  np.reshape(sample, (img_rows, img_cols, nchannels)),
                #  np.reshape(adv_x, (img_rows, img_cols, nchannels)), figure)



            results[target, sample_ind] = res
            perturbations[target, sample_ind] = percent_perturb

    print('--------------------------------------')
    #print("results:", results)

    # Compute the number of adversarial examples that were successfully found
    nb_targets_tried = ((nb_classes - 1) * source_samples)
    succ_rate = float(np.sum(results)) / nb_targets_tried
    #print(np.sum(results), nb_targets_tried)
    #sys.exit()
    print('Avg. rate of successful adv. examples {0:.4f}'.format(succ_rate))
    report.clean_train_adv_eval = 1. - succ_rate

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(perturbations)
    print('Avg. rate of perturbed features {0:.4f}'.format(percent_perturbed))

    # Compute the average distortion introduced for successful samples only
    percent_perturb_succ = np.mean(perturbations * (results == 1))
    print('Avg. rate of perturbed features for successful '
        'adversarial examples {0:.4f}'.format(percent_perturb_succ))

    # Close TF session
    sess.close()













    #img_rows, img_cols = x_test.shape[1], x_test.shape[2]
    #nb_classes = y_test.shape[1]

    ### Label smoothing (Not really sure why we'd want to do this [changes 0.0,1.0 to 0.05,0.95))
    ##label_smoothing = 0.1
    ##y_train -= label_smoothing * (y_train - 1. / nb_classes)

    ## Define Keras model
    #model = ModelFuncPointer(img_rows=img_rows, img_cols=img_cols,
    #                channels=None, nb_filters=None,
    #                nb_classes=nb_classes)
    #print("Defined Keras model.")

    ## To be able to call the model in the custom loss, we need to call it once
    #model(model.input)

    ## Load weights if given
    #if(init != None):
    #    model.load_weights(init)

    ## Initialize the Fast Gradient Sign Method (FGSM) attack object
    #wrap = KerasModelWrapper(model)
    #attack = FastGradientMethod(wrap, sess=sess)
    ##attack = LBFGS(wrap, sess=sess)
    ##attack = SPSA(wrap, sess=sess)
    ##attack = SaliencyMapMethod(wrap, sess=sess)
    #attack_params = {'eps': 1.,
    #             'clip_min': 0.,
    #             'clip_max': 1.}
    ##attack_params = {'batch_size': 64,
    ##            'clip_min': 0.,
    ##            'clip_max': 1.}
    ##attack_params = {'theta': 1., 'gamma': 0.1,
    ##             'clip_min': 0., 'clip_max': 1.,
    ##             'y_target': None}

    #adv_acc_metric = get_adversarial_acc_metric(model, attack, attack_params)
    #model.compile(
    #  optimizer=keras.optimizers.Adam(learningRate),
    #  loss='categorical_crossentropy',
    #  metrics=['accuracy', adv_acc_metric]
    #)

    ## Early stopping and saving the best weights
    #callbacks_list = [
    #        keras.callbacks.EarlyStopping(
    #            verbose=1,
    #            min_delta=0.01,
    #            patience=25),
    #        keras.callbacks.ModelCheckpoint(
    #            filepath=network[1],
    #            save_best_only=True)
    #        ]

    #history = model.fit_generator(TrainGenerator,
    #    steps_per_epoch= epochSteps,
    #    epochs=numEpochs,
    #    validation_data=ValidationGenerator,
    #    validation_steps=validationSteps,
    #    use_multiprocessing=True,
    #    callbacks=callbacks_list,
    #    max_queue_size=nCPU,
    #    workers=nCPU,
    #    )


    ## Save genotype images for testset
    #print("\nGenerating adversarial examples and writing images/predicions...")
    #imageDir = os.path.join(projectDir,"test_images")
    #if not os.path.exists(imageDir):
    #    os.makedirs(imageDir)
    #for i in range(x_test.shape[0]):
    #    org_predFILE = os.path.join(imageDir,"examp{}_org_pred.txt".format(i))
    #    adv_predFILE = os.path.join(imageDir,"examp{}_adv_pred.txt".format(i))
    #    org_gmFILE = os.path.join(imageDir,"examp{}_org.npy".format(i))
    #    adv_gmFILE = os.path.join(imageDir,"examp{}_adv.npy".format(i))
    #    org_imageFILE = os.path.join(imageDir,"examp{}_org.png".format(i))
    #    adv_imageFILE = os.path.join(imageDir,"examp{}_adv.png".format(i))
    #    delta_imageFILE = os.path.join(imageDir,"examp{}_delta.png".format(i))
    #    org_example = x_test[i].reshape((1, x_test.shape[1], x_test.shape[2]))
    #    adv_example = attack.generate_np(org_example, **attack_params)
    #    pred_org = model.predict(org_example)
    #    pred_adv = model.predict(adv_example)
    #    np.savetxt(org_predFILE, pred_org)
    #    np.savetxt(adv_predFILE, pred_adv)
    #    org_image = org_example.reshape((img_rows, img_cols))
    #    adv_image = adv_example.reshape((img_rows, img_cols))
    #    delta_image = org_image - adv_image
    #    np.save(org_gmFILE, org_example)
    #    np.save(adv_gmFILE, adv_example)
    #    plt.imsave(org_imageFILE, org_image)
    #    plt.imsave(adv_imageFILE, adv_image)
    #    plt.imsave(delta_imageFILE, delta_image)
    #    progress_bar(i/float(x_test.shape[0]))

    #print("completed_1")
    #sys.exit()

    ## Evaluate the accuracy on legitimate and adversarial test examples
    #report = AccuracyReport()
    #_, acc, adv_acc = model.evaluate(x_test, y_test,
    #                               batch_size=64,
    #                               verbose=0)
    #report.clean_train_clean_eval = acc
    #report.clean_train_adv_eval = adv_acc
    #outLog = resultsFile.replace(".p","_log.txt")
    #with open(outLog, "w") as fOUT:
    #    fOUT.write("Before adversarial training\n")
    #    fOUT.write("===========================\n")
    #    fOUT.write("Test accuracy on legitimate examples: %0.4f\n" % acc)
    #    fOUT.write("Test accuracy on adversarial examples: %0.4f\n" % adv_acc)
    #print('\nTest accuracy on legitimate examples: %0.4f' % acc)
    #print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

    ## Write the network
    #if(network != None):
    #    ##serialize model to JSON
    #    model_json = model.to_json()
    #    with open(network[0], "w") as json_file:
    #        json_file.write(model_json)

    ## Load json and create model
    #if(network != None):
    #    jsonFILE = open(network[0],"r")
    #    loadedModel = jsonFILE.read()
    #    jsonFILE.close()
    #    model=model_from_json(loadedModel)
    #    model.load_weights(network[1])
    #else:
    #    print("Error: model and weights not loaded")
    #    sys.exit(1)

    #predictions = model.predict(x_test)

    #history.history['loss'] = np.array(history.history['loss'])
    #history.history['val_loss'] = np.array(history.history['val_loss'])
    #history.history['predictions'] = np.array(predictions)
    #history.history['Y_test'] = np.array(y_test)
    #history.history['name'] = ModelName

    #print("results written to: ",resultsFile)
    #pickle.dump(history.history, open( resultsFile, "wb" ))



    ########### Adversarial training #############
    #print("Repeating the process, using adversarial training")
    ## Redefine Keras model
    #model_2 = ModelFuncPointer(img_rows=img_rows, img_cols=img_cols,
    #                channels=None, nb_filters=None,
    #                nb_classes=nb_classes)
    #model_2(model_2.input)
    #wrap_2 = KerasModelWrapper(model_2)
    #attack_2 = FastGradientMethod(wrap_2, sess=sess)
    ##attack_2 = LBFGS(wrap_2, sess=sess)


    ## Use a loss function based on legitimate and adversarial examples
    #adv_loss_2 = get_adversarial_loss(model_2, attack_2, attack_params)
    #adv_acc_metric_2 = get_adversarial_acc_metric(model_2, attack_2, attack_params)

    #model_2.compile(
    #  optimizer=keras.optimizers.Adam(learningRate),
    #  loss=adv_loss_2,
    #  metrics=['accuracy', adv_acc_metric_2]
    #)

    ## Early stopping and saving the best weights
    #callbacks_list_2 = [
    #        keras.callbacks.EarlyStopping(
    #            verbose=1,
    #            min_delta=0.01,
    #            patience=25),
    #        keras.callbacks.ModelCheckpoint(
    #            filepath=network[1].replace(".h5","_2.h5"),
    #            save_best_only=True)
    #        ]

    ## Train model
    #history_2 = model_2.fit_generator(TrainGenerator,
    #    steps_per_epoch= epochSteps,
    #    epochs=numEpochs,
    #    validation_data=ValidationGenerator,
    #    validation_steps=validationSteps,
    #    use_multiprocessing=True,
    #    callbacks=callbacks_list_2,
    #    max_queue_size=nCPU,
    #    workers=nCPU,
    #    )

    ## Evaluate the accuracy on legitimate and adversarial test examples
    #_, acc, adv_acc = model_2.evaluate(x_test, y_test,
    #                                 batch_size=64,
    #                                 verbose=0)

    #report.adv_train_clean_eval = acc
    #report.adv_train_adv_eval = adv_acc
    #with open(outLog, "a") as fOUT:
    #    fOUT.write("\nAfter adversarial training\n")
    #    fOUT.write("===========================\n")
    #    fOUT.write("Test accuracy on legitimate examples: %0.4f\n" % acc)
    #    fOUT.write("Test accuracy on adversarial examples: %0.4f\n" % adv_acc)
    #print('Test accuracy on legitimate examples: %0.4f' % acc)
    #print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)


    ## Write the network
    #if(network != None):
    #    ##serialize model_2 to JSON
    #    model_json_2 = model_2.to_json()
    #    with open(network[0].replace(".json","_2.json"), "w") as json_file:
    #        json_file.write(model_json_2)

    ## Load json and create model
    #if(network != None):
    #    jsonFILE = open(network[0].replace(".json","_2.json"),"r")
    #    loadedModel_2 = jsonFILE.read()
    #    jsonFILE.close()
    #    model_2=model_from_json(loadedModel_2)
    #    model_2.load_weights(network[1].replace(".h5","_2.h5"))
    #else:
    #    print("Error: model_2 and weights_2 not loaded")
    #    sys.exit(1)

    #predictions_2 = model_2.predict(x_test)

    #history_2.history['loss'] = np.array(history_2.history['loss'])
    #history_2.history['val_loss'] = np.array(history_2.history['val_loss'])
    #history_2.history['predictions'] = np.array(predictions_2)
    #history_2.history['Y_test'] = np.array(y_test)
    #history_2.history['name'] = ModelName+"_adversarialTrained"

    #print("results written to: ",resultsFile.replace(".p","_2.p"))
    #pickle.dump(history_2.history, open( resultsFile.replace(".p","_2.p"), "wb" ))

    #reportFile = os.path.join(NetworkDir, "testReport.p")
    #print("report written to: ",reportFile)
    #pickle.dump(report, open( reportFile, "wb" ))

    #return None

#-------------------------------------------------------------------------------------------

def runModelsMisspecified(ModelFuncPointer,
            ModelName,
            NetworkDir,
            projectDir,
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
            learningRate=0.001
            ):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuID)

    # Force TensorFlow to use single thread to improve reproducibility
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                          inter_op_parallelism_threads=1)

    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    if(resultsFile == None):
        resultsFilename = os.path.basename(trainFile)[:-4] + ".p"
        resultsFile = os.path.join("./results/",resultsFilename)

    # Read all test data into memory
    x_test, y_test = TestGenerator.__getitem__(0)
    img_rows, img_cols = x_test.shape[1], x_test.shape[2]
    nb_classes = y_test.shape[1]

    # Save images/post-generator genotypes for testset
    print("\nWriting images and post-generator genotypes...")
    imageDir = os.path.join(projectDir,"test_images")
    if not os.path.exists(imageDir):
        os.makedirs(imageDir)
    for i in range(x_test.shape[0]):
        org_gmFILE = os.path.join(imageDir,"examp{}_org.npy".format(i))
        org_imageFILE = os.path.join(imageDir,"examp{}_org.png".format(i))
        org_example = x_test[i].reshape((1, x_test.shape[1], x_test.shape[2]))
        org_image = org_example.reshape((img_rows, img_cols))
        np.save(org_gmFILE, org_example)
        plt.imsave(org_imageFILE, org_image)
        progress_bar(i/float(x_test.shape[0]))

    # Load json and create model
    if(network != None):
        jsonFILE = open(network[0],"r")
        loadedModel = jsonFILE.read()
        jsonFILE.close()
        model=model_from_json(loadedModel)
        model.load_weights(network[1])
    else:
        print("Error: model and weights not loaded")
        sys.exit(1)

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
    print("new results written to: ",resultsFile)
    pickle.dump(history, open(resultsFile, "wb"))


    ########### Adversarial training #############
    # Load json and create model
    if(network != None):
        jsonFILE = open(network[0].replace(".json","_2.json"),"r")
        loadedModel_2 = jsonFILE.read()
        jsonFILE.close()
        model_2=model_from_json(loadedModel_2)
        model_2.load_weights(network[1].replace(".h5","_2.h5"))
    else:
        print("Error: model_2 and weights_2 not loaded")
        sys.exit(1)

    predictions_2 = model_2.predict(x_test)

    #replace predictions and T_test in results file 2
    history_2 = pickle.load(open(resultsFile.replace(".p","_2.p"), "rb"))
    tmp = []
    for gr in test_info["gr"]:
        if gr > 0.0:
            tmp.append([0.0,1.0])
        else:
            tmp.append([1.0,0.0])
    history_2["Y_test"] = np.array(tmp)
    history_2['predictions'] = np.array(predictions_2)

    # rewrite new results file
    print("new results written to: ",resultsFile.replace(".p","_2.p"))
    pickle.dump(history_2, open(resultsFile.replace(".p","_2.p"), "wb"))

    return None

#-------------------------------------------------------------------------------------------

def progress_bar(percent, barLen = 50):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()

#-------------------------------------------------------------------------------------------

def get_adversarial_acc_metric(model, attack, attack_params):
  def adv_acc(y, _):
    # Generate adversarial examples
    x_adv = attack.generate(model.input, **attack_params)
    # Consider the attack to be constant
    x_adv = tf.stop_gradient(x_adv)

    # Accuracy on the adversarial examples
    preds_adv = model(x_adv)
    return keras.metrics.categorical_accuracy(y, preds_adv)

  return adv_acc

#-------------------------------------------------------------------------------------------

def get_adversarial_loss(model, attack, attack_params):
  def adv_loss(y, preds):
    # Cross-entropy on the legitimate examples
    cross_ent = keras.losses.categorical_crossentropy(y, preds)

    # Generate adversarial examples
    x_adv = attack.generate(model.input, **attack_params)
    # Consider the attack to be constant
    x_adv = tf.stop_gradient(x_adv)

    # Cross-entropy on the adversarial examples
    preds_adv = model(x_adv)
    cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

    return 0.5 * cross_ent + 0.5 * cross_ent_adv

  return adv_loss

#-------------------------------------------------------------------------------------------

def indicesGenerator(batchSize,numReps):
    '''
    Generate indices randomly from range (0,numReps) in batches of size batchSize
    without replacement.

    This is for the batch generator to randomly choose trees from a directory
    but make sure
    '''
    availableIndices = np.arange(numReps)
    np.random.shuffle(availableIndices)
    ci = 0
    while 1:
        if((ci+batchSize) > numReps):
            ci = 0
            np.random.shuffle(availableIndices)
        batchIndices = availableIndices[ci:ci+batchSize]
        ci = ci+batchSize

        yield batchIndices

#-------------------------------------------------------------------------------------------

def getHapsPosLabels(direc,simulator,shuffle=False):
    '''
    loops through a trees directory created by the data generator class
    and returns the repsective genotype matrices, positions, and labels
    '''
    haps = []
    positions = []
    infoFilename = os.path.join(direc,"info.p")
    infoDict = pickle.load(open(infoFilename,"rb"))
    labels = infoDict["y"]

    #how many trees files are in this directory.
    li = os.listdir(direc)
    numReps = len(li) - 1   #minus one for the 'info.p' file

    for i in range(numReps):
        filename = str(i) + ".trees"
        filepath = os.path.join(direc,filename)
        treeSequence = msp.load(filepath)
        haps.append(treeSequence.genotype_matrix())
        positions.append(np.array([s.position for s in treeSequence.sites()]))


    haps = np.array(haps)
    positions = np.array(positions)

    return haps,positions,labels

#-------------------------------------------------------------------------------------------

def simplifyTreeSequenceOnSubSampleSet_stub(ts,numSamples):
    '''
    This function should take in a tree sequence, generate
    a subset the size of numSamples, and return the tree sequence simplified on
    that subset of individuals
    '''

    ts = ts.simplify() #is this neccessary
    inds = [ind.id for ind in ts.individuals()]
    sample_subset = np.sort(np.random.choice(inds,sample_size,replace=False))
    sample_nodes = []
    for i in sample_subset:
        ind = ts.individual(i)
        sample_nodes.append(ind.nodes[0])
        sample_nodes.append(ind.nodes[1])

    ts = ts.simplify(sample_nodes)

    return ts

#-------------------------------------------------------------------------------------------

def sort_min_diff(amat):
    '''this function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted by genetic similarity.
    this problem is NP, so here we use a nearest neighbors approx.  it's not perfect, but it's fast and generally performs ok.
    assumes your input matrix is a numpy array'''

    mb = NearestNeighbors(len(amat), metric='manhattan').fit(amat)
    v = mb.kneighbors(amat)
    smallest = np.argmin(v[0].sum(axis=1))
    return amat[v[1][smallest]]

#-------------------------------------------------------------------------------------------

def mutateTrees(treesDirec,outputDirec,muLow,muHigh,numMutsPerTree=1,simulator="msprime"):
    '''
    read in .trees files from treesDirec, mutate that tree numMuts seperate times
    using a mutation rate pulled from a uniform dirstribution between muLow and muHigh

    also, re-write the labels file to reflect.
    '''
    if(numMutsPerTree > 1):
        assert(treesDirec != outputDirec)

    if not os.path.exists(outputDirec):
        print("directory '",outputDirec,"' does not exist, creating it")
        os.makedirs(outputDirec)

    infoFilename = os.path.join(treesDirec,"info.p")
    infoDict = pickle.load(open(infoFilename,"rb"))
    labels = infoDict["y"]

    newLabels = []
    newMaxSegSites = 0

    #how many trees files are in this directory.
    li = os.listdir(treesDirec)
    numReps = len(li) - 1   #minus one for the 'labels.txt' file

    for i in range(numReps):
        filename = str(i) + ".trees"
        filepath = os.path.join(treesDirec,filename)
        treeSequence = msp.load(filepath)
        blankTreeSequence = msp.mutate(treeSequence,0)
        rho = labels[i]
        for mut in range(numMuts):
            simNum = (i*numMuts) + mut
            simFileName = os.path.join(outputDirec,str(simNum)+".trees")
            mutationRate = np.random.uniform(muLow,muHigh)
            mutatedTreeSequence = msp.mutate(blankTreeSequence,mutationRate)
            mutatedTreeSequence.dump(simFileName)
            newMaxSegSites = max(newMaxSegSites,mutatedTreeSequence.num_sites)
            newLabels.append(rho)

    infoCopy = copy.deepcopy(infoDict)
    infoCopy["maxSegSites"] = newMaxSeqSites
    if(numMutsPerTree > 1):
        infoCopy["y"] = np.array(newLabels,dtype="float32")
        infoCopy["numReps"] = numReps * numMuts
    outInfoFilename = os.path.join(outputDirec,"info.p")
    pickle.dump(infocopy,open(outInfoFilename,"wb"))

    return None

#-------------------------------------------------------------------------------------------

def segSitesStats(treesDirec):
    '''
    DEPRICATED
    '''

    infoFilename = os.path.join(treesDirec,"info.p")
    infoDict = pickle.load(open(infoFilename,"rb"))

    newLabels = []
    newMaxSegSites = 0

    #how many trees files are in this directory.
    li = os.listdir(treesDirec)
    numReps = len(li) - 1   #minus one for the 'labels.txt' file

    segSites = []

    for i in range(numReps):
        filename = str(i) + ".trees"
        filepath = os.path.join(treesDirec,filename)
        treeSequence = msp.load(filepath)
        segSites.append(treeSequence.num_sites)

    return segSites

#-------------------------------------------------------------------------------------------

def mae(x,y):
    '''
    Compute mean absolute error between predictions and targets

    float[],float[] -> float
    '''
    assert(len(x) == len(y))
    summ = 0.0
    length = len(x)
    for i in range(length):
        summ += abs(x[i] - y[i])
    return summ/length

#-------------------------------------------------------------------------------------------

def mse(x,y):
    '''
    Compute mean squared error between predictions and targets

    float[],float[] -> float
    '''

    assert(len(x) == len(y))
    summ = 0.0
    length = len(x)
    for i in range(length):
        summ += (x[i] - y[i])**2
    return summ/length

#-------------------------------------------------------------------------------------------

def plotResults(resultsFile,saveas):

    '''
    plotting code for testing a model on simulation.
    using the resulting pickle file on a training run (resultsFile).
    This function plots the results of the final test set predictions,
    as well as validation loss as a function of Epochs during training.

    '''

    plt.rc('font', family='serif', serif='Times')
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)
    plt.rc('axes', labelsize=6)

    results = pickle.load(open( resultsFile , "rb" ))

    fig,axes = plt.subplots(2,1)
    plt.subplots_adjust(hspace=0.5)

    predictions = np.array([float(Y) for Y in results["predictions"]])
    realValues = np.array([float(X) for X in results["Y_test"]])

    r_2 = round((np.corrcoef(predictions,realValues)[0,1])**2,5)

    mae_0 = round(mae(realValues,predictions),4)
    mse_0 = round(mse(realValues,predictions),4)
    labels = "$R^{2} = $"+str(r_2)+"\n"+"$mae = $" + str(mae_0)+" | "+"$mse = $" + str(mse_0)

    axes[0].scatter(realValues,predictions,marker = "o", color = 'tab:purple',s=5.0,alpha=0.6)

    lims = [
        np.min([axes[0].get_xlim(), axes[0].get_ylim()]),  # min of both axes
        np.max([axes[0].get_xlim(), axes[0].get_ylim()]),  # max of both axes
    ]
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    axes[0].set_title(results["name"]+"\n"+labels,fontsize=6)

    lossRowIndex = 1
    axes[1].plot(results["loss"],label = "mae loss",color='tab:cyan')
    axes[1].plot(results["val_loss"], label= "mae validation loss",color='tab:pink')

    #axes[1].plot(results["mean_squared_error"],label = "mse loss",color='tab:green')
    #axes[1].plot(results["val_mean_squared_error"], label= "mse validation loss",color='tab:olive')

    axes[1].legend(frameon = False,fontsize = 6)
    axes[1].set_ylabel("mse")

    axes[0].set_ylabel(str(len(predictions))+" msprime predictions")
    axes[0].set_xlabel(str(len(realValues))+" msprime real values")
    fig.subplots_adjust(left=.15, bottom=.16, right=.85, top=.92,hspace = 0.5,wspace=0.4)
    height = 7.00
    width = 7.00

    axes[0].grid()
    fig.set_size_inches(height, width)
    fig.savefig(saveas)

#-------------------------------------------------------------------------------------------

def plotResultsSigmoid(resultsFile,saveas):

    plt.rc('font', family='serif', serif='Times')
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)
    plt.rc('axes', labelsize=6)

    results = pickle.load(open( resultsFile , "rb" ))

    fig,axes = plt.subplots(2,1)
    plt.subplots_adjust(hspace=0.5)

    predictions = np.array([float(Y) for Y in results["predictions"]])
    realValues = np.array([float(X) for X in results["Y_test"]])

    const, expan = [], []
    for i, val in enumerate(realValues):
        if val == 0:
            const.append(predictions[i])
        else:
            expan.append(predictions[i])
    np.array(const)
    np.array(expan)

    mae_0 = round(mae(realValues,predictions),4)
    mse_0 = round(mse(realValues,predictions),4)
    labels = "$mae = $" + str(mae_0)+" | "+"$mse = $" + str(mse_0)

    n_bins = np.linspace(0.0,1.0,100)
    axes[0].hist(const, n_bins, color="orange", label="Constant size", alpha=0.5)
    axes[0].hist(expan, n_bins, color="blue", label="Exponential growth", alpha=0.5)
    axes[0].axvline(x=0.5, linestyle="--", linewidth=0.3, color="black")
    axes[0].legend(prop={'size': 4})

    lossRowIndex = 1
    axes[1].plot(results["loss"],label = "mae loss",color="orange")
    axes[1].plot(results["val_loss"], label= "mae validation loss",color="blue")

    axes[1].legend(frameon = False,fontsize = 6)
    axes[1].set_ylabel("mse")

    axes[0].set_ylabel("N")
    axes[0].set_xlabel("sigmoid output")
    fig.subplots_adjust(left=.15, bottom=.16, right=.85, top=.92,hspace = 0.5,wspace=0.4)
    height = 4.00
    width = 4.00

    fig.set_size_inches(height, width)
    fig.savefig(saveas)

#-------------------------------------------------------------------------------------------

def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

#-------------------------------------------------------------------------------------------

def plotResultsSoftmax2(resultsFile,saveas):

    plt.rc('font', family='serif', serif='Times')
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)
    plt.rc('axes', labelsize=6)

    results = pickle.load(open( resultsFile , "rb" ))

    fig,axes = plt.subplots(2,1)
    plt.subplots_adjust(hspace=0.5)

    predictions = results["predictions"]
    realValues = results["Y_test"]

    const, expan = [], []
    for i, val in enumerate(realValues):
        if val[0] == 1.0:
            const.append(1.0 - predictions[i][0])
        else:
            expan.append(predictions[i][1])
    np.array(const)
    np.array(expan)

    ce = round(cross_entropy(predictions,realValues),4)
    labels = "Cross entropy = " + str(ce)

    n_bins = np.linspace(0.0,1.0,100)
    axes[0].hist(const, n_bins, color="orange", label="Constant size", alpha=0.5)
    axes[0].hist(expan, n_bins, color="blue", label="Exponential growth", alpha=0.5)
    axes[0].axvline(x=0.5, linestyle="--", linewidth=0.4, color="black")
    axes[0].legend(prop={'size': 4})
    axes[0].set_title(results["name"]+"\n"+labels,fontsize=6)

    lossRowIndex = 1
    axes[1].plot(results["loss"],label = "CE loss",color="orange")
    axes[1].plot(results["val_loss"], label= "CE validation loss",color="blue")

    axes[1].legend(frameon = False,fontsize = 6)
    axes[1].set_ylabel("Cross entropy")

    axes[0].set_ylabel("N")
    axes[0].set_xlabel("Predicted probability of exponential growth")
    fig.subplots_adjust(left=.15, bottom=.16, right=.85, top=.92,hspace = 0.5,wspace=0.4)
    height = 4.00
    width = 4.00

    fig.set_size_inches(height, width)
    fig.savefig(saveas)

#-------------------------------------------------------------------------------------------

def plotResultsSoftmax2Heatmap(resultsFile,saveas):

    plt.rc('font', family='serif', serif='Times')
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)
    plt.rc('axes', labelsize=6)

    results = pickle.load(open( resultsFile , "rb" ))

    fig,axes = plt.subplots(2,1)
    plt.subplots_adjust(hspace=0.5)

    predictions = results["predictions"]
    realValues = results["Y_test"]

    const, expan = [], []

    const_const, const_expan, expan_const, expan_expan = 0,0,0,0
    const_total, expan_total = 0,0
    for i, val in enumerate(realValues):
        if val[0] == 1.0:
            const_total+=1
            const.append(1.0 - predictions[i][0])
            if predictions[i][0] > 0.5:
                const_const+=1
            if predictions[i][1] > 0.5:
                const_expan+=1
        else:
            expan_total+=1
            expan.append(predictions[i][1])
            if predictions[i][0] > 0.5:
                expan_const+=1
            if predictions[i][1] > 0.5:
                expan_expan+=1
    np.array(const)
    np.array(expan)

    ce = round(cross_entropy(predictions,realValues),4)
    labels = "Cross entropy = " + str(ce)

    data=np.array([[const_const/float(const_total),const_expan/float(const_total)],[expan_const/float(expan_total),expan_expan/float(expan_total)]])
    rowLabels = ["Constant", "Growth"]
    heatmap = axes[0].pcolor(data, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(heatmap, cmap=plt.cm.Blues, ax=axes[0])
    cbar.set_label('Proportion assigned to class', rotation=270, labelpad=20)

    # put the major ticks at the middle of each cell
    axes[0].set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    axes[0].set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    axes[0].invert_yaxis()
    axes[0].xaxis.tick_top()

    plt.tick_params(axis='y', which='both', right='off')
    plt.tick_params(axis='x', which='both', direction='out')
    axes[0].set_xticklabels(["Constant", "Growth"], minor=False, fontsize=6)


    axes[0].set_yticklabels(rowLabels, minor=False, fontsize=6)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            val = data[y, x]
            val *= 100
            if val > 50:
                c = '0.9'
            else:
                c = 'black'
            axes[0].text(x + 0.5, y + 0.5, '%.1f%%' % val, horizontalalignment='center', verticalalignment='center', color=c, fontsize=6)


    axes[0].set_title(results["name"]+"\n"+labels,fontsize=6)

    lossRowIndex = 1
    axes[1].plot(results["loss"],label = "CE loss",color="orange")
    axes[1].plot(results["val_loss"], label= "CE validation loss",color="blue")

    axes[1].legend(frameon = False,fontsize = 6)
    axes[1].set_ylabel("Cross entropy")

    fig.subplots_adjust(left=.15, bottom=.15, right=.85, top=0.87, hspace = 0.5, wspace=0.4)
    height = 4.00
    width = 4.00

    fig.set_size_inches(height, width)
    fig.savefig(saveas)

#-------------------------------------------------------------------------------------------

def plotResultsSoftmax2HeatmapMis(resultsFile, resultsFile2, saveas):

    plt.rc('font', family='serif', serif='Times')
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)
    plt.rc('axes', labelsize=6)

    results = pickle.load(open( resultsFile , "rb" ))

    fig,axes = plt.subplots(2,1)
    plt.subplots_adjust(hspace=0.5)

    predictions = results["predictions"]
    realValues = results["Y_test"]

    const, expan = [], []

    const_const, const_expan, expan_const, expan_expan = 0,0,0,0
    const_total, expan_total = 0,0
    for i, val in enumerate(realValues):
        if val[0] == 1.0:
            const_total+=1
            const.append(1.0 - predictions[i][0])
            if predictions[i][0] > 0.5:
                const_const+=1
            if predictions[i][1] > 0.5:
                const_expan+=1
        else:
            expan_total+=1
            expan.append(predictions[i][1])
            if predictions[i][0] > 0.5:
                expan_const+=1
            if predictions[i][1] > 0.5:
                expan_expan+=1
    np.array(const)
    np.array(expan)

    ce = round(cross_entropy(predictions,realValues),4)
    labels = "Cross entropy = " + str(ce)

    data=np.array([[const_const/float(const_total),const_expan/float(const_total)],[expan_const/float(expan_total),expan_expan/float(expan_total)]])
    rowLabels = ["Constant", "Growth"]
    heatmap = axes[0].pcolor(data, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(heatmap, cmap=plt.cm.Blues, ax=axes[0])
    cbar.set_label('Proportion assigned to class', rotation=270, labelpad=20)

    # put the major ticks at the middle of each cell
    axes[0].set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    axes[0].set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    axes[0].invert_yaxis()
    axes[0].xaxis.tick_top()

    plt.tick_params(axis='y', which='both', right='off')
    plt.tick_params(axis='x', which='both', direction='out')
    axes[0].set_xticklabels(["Constant", "Growth"], minor=False, fontsize=6)


    axes[0].set_yticklabels(rowLabels, minor=False, fontsize=6)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            val = data[y, x]
            val *= 100
            if val > 50:
                c = '0.9'
            else:
                c = 'black'
            axes[0].text(x + 0.5, y + 0.5, '%.1f%%' % val, horizontalalignment='center', verticalalignment='center', color=c, fontsize=6)


    axes[0].set_title(results["name"]+"\n"+labels,fontsize=6)

    results = pickle.load(open(resultsFile2 , "rb"))

    predictions = results["predictions"]
    realValues = results["Y_test"]

    const, expan = [], []

    const_const, const_expan, expan_const, expan_expan = 0,0,0,0
    const_total, expan_total = 0,0
    for i, val in enumerate(realValues):
        if val[0] == 1.0:
            const_total+=1
            const.append(1.0 - predictions[i][0])
            if predictions[i][0] > 0.5:
                const_const+=1
            if predictions[i][1] > 0.5:
                const_expan+=1
        else:
            expan_total+=1
            expan.append(predictions[i][1])
            if predictions[i][0] > 0.5:
                expan_const+=1
            if predictions[i][1] > 0.5:
                expan_expan+=1
    np.array(const)
    np.array(expan)

    ce = round(cross_entropy(predictions,realValues),4)
    labels = "Cross entropy = " + str(ce)

    data=np.array([[const_const/float(const_total),const_expan/float(const_total)],[expan_const/float(expan_total),expan_expan/float(expan_total)]])
    rowLabels = ["Constant", "Growth"]
    heatmap = axes[1].pcolor(data, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(heatmap, cmap=plt.cm.Blues, ax=axes[1])
    cbar.set_label('Proportion assigned to class', rotation=270, labelpad=20)

    # put the major ticks at the middle of each cell
    axes[1].set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    axes[1].set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    axes[1].invert_yaxis()
    axes[1].xaxis.tick_top()

    plt.tick_params(axis='y', which='both', right='off')
    plt.tick_params(axis='x', which='both', direction='out')
    axes[1].set_xticklabels(["Constant", "Growth"], minor=False, fontsize=6)


    axes[1].set_yticklabels(rowLabels, minor=False, fontsize=6)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            val = data[y, x]
            val *= 100
            if val > 50:
                c = '0.9'
            else:
                c = 'black'
            axes[1].text(x + 0.5, y + 0.5, '%.1f%%' % val, horizontalalignment='center', verticalalignment='center', color=c, fontsize=6)


    axes[1].set_title(results["name"]+"\n"+labels,fontsize=6)

    fig.subplots_adjust(left=.15, bottom=.05, right=.85, top=0.87, hspace = 0.6, wspace=0.4)
    height = 4.00
    width = 4.00

    fig.set_size_inches(height, width)
    fig.savefig(saveas)

#-------------------------------------------------------------------------------------------

def plotSummaryStats(projectDir_A, projectDir_B, saveas):

    ## Load all test results
    test_info_A = pickle.load(open(os.path.join(projectDir_A, "test", "info.p"), "rb"))
    test_info_B = pickle.load(open(os.path.join(projectDir_B, "test", "info.p"), "rb"))
    G_A_org, G_A_adv, G_B_org = [],[],[]
    for i in range(test_info_A["numReps"]):
        Hfilepath = os.path.join(projectDir_A, "test_images", "examp%s_org.npy" %(i))
        H = np.load(Hfilepath)
        G_A_org.append(H[0])
        Hfilepath = os.path.join(projectDir_A, "test_images", "examp%s_adv.npy" %(i))
        H = np.load(Hfilepath)
        G_A_adv.append(H[0])
        Hfilepath = os.path.join(projectDir_B, "test_images", "examp%s_org.npy" %(i))
        H = np.load(Hfilepath)
        G_B_org.append(H[0])
    G_A_org = np.array(G_A_org,dtype="int8")
    G_A_adv = np.array(G_A_adv,dtype="int8")
    G_B_org = np.array(G_B_org,dtype="int8")

    ## Calculate stats for projectDir_A original examples
    A_org_ng_D, A_org_gr_D = [], []
    for i, gm in enumerate(G_A_org):
        haps = allel.HaplotypeArray(gm)
        gens = allel.GenotypeArray(haps.to_genotypes(ploidy=2))
        ac = gens.count_alleles()
        D = allel.tajima_d(ac)
        if test_info_A["gr"][i] > 0.0:
            A_org_gr_D.append(D)
        else:
            A_org_ng_D.append(D)
    print("A_org_ng_D:", np.average(np.array(A_org_ng_D)))
    print("A_org_gr_D:", np.average(np.array(A_org_gr_D)))
    print("=============================================")
    print("=============================================")

    ## Calculate stats for projectDir_A adversarial examples
    A_adv_ng_D, A_adv_gr_D = [], []
    for i, gm in enumerate(G_A_adv):
        haps = allel.HaplotypeArray(gm)
        gens = allel.GenotypeArray(haps.to_genotypes(ploidy=2))
        ac = gens.count_alleles()
        D = allel.tajima_d(ac)
        if test_info_A["gr"][i] > 0.0:
            A_adv_gr_D.append(D)
        else:
            A_adv_ng_D.append(D)
    print("A_adv_ng_D:", np.average(np.array(A_adv_ng_D)))
    print("A_adv_gr_D:", np.average(np.array(A_adv_gr_D)))
    print("=============================================")
    print("=============================================")

    ## Calculate stats for projectDir_B original examples
    B_org_ng_D, B_org_gr_D = [], []
    for i, gm in enumerate(G_B_org):
        haps = allel.HaplotypeArray(gm)
        gens = allel.GenotypeArray(haps.to_genotypes(ploidy=2))
        ac = gens.count_alleles()
        D = allel.tajima_d(ac)
        if test_info_B["gr"][i] > 0.0:
            B_org_gr_D.append(D)
        else:
            B_org_ng_D.append(D)
    print("B_org_ng_D:", np.average(np.array(B_org_ng_D)))
    print("B_org_gr_D:", np.average(np.array(B_org_gr_D)))
    print("=============================================")
    print("=============================================")
    #plt.rc('font', family='serif', serif='Times')
    #plt.rc('xtick', labelsize=6)
    #plt.rc('ytick', labelsize=6)
    #plt.rc('axes', labelsize=6)

    #results = pickle.load(open( resultsFile , "rb" ))

    #fig,axes = plt.subplots(2,1)
    #plt.subplots_adjust(hspace=0.5)

    #predictions = results["predictions"]
    #realValues = results["Y_test"]

    #const, expan = [], []

    #const_const, const_expan, expan_const, expan_expan = 0,0,0,0
    #const_total, expan_total = 0,0
    #for i, val in enumerate(realValues):
    #    if val[0] == 1.0:
    #        const_total+=1
    #        const.append(1.0 - predictions[i][0])
    #        if predictions[i][0] > 0.5:
    #            const_const+=1
    #        if predictions[i][1] > 0.5:
    #            const_expan+=1
    #    else:
    #        expan_total+=1
    #        expan.append(predictions[i][1])
    #        if predictions[i][0] > 0.5:
    #            expan_const+=1
    #        if predictions[i][1] > 0.5:
    #            expan_expan+=1
    #np.array(const)
    #np.array(expan)

    #ce = round(cross_entropy(predictions,realValues),4)
    #labels = "Cross entropy = " + str(ce)

    #data=np.array([[const_const/float(const_total),const_expan/float(const_total)],[expan_const/float(expan_total),expan_expan/float(expan_total)]])
    #rowLabels = ["Constant", "Growth"]
    #heatmap = axes[0].pcolor(data, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    #cbar = plt.colorbar(heatmap, cmap=plt.cm.Blues, ax=axes[0])
    #cbar.set_label('Proportion assigned to class', rotation=270, labelpad=20)

    ## put the major ticks at the middle of each cell
    #axes[0].set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    #axes[0].set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    #axes[0].invert_yaxis()
    #axes[0].xaxis.tick_top()

    #plt.tick_params(axis='y', which='both', right='off')
    #plt.tick_params(axis='x', which='both', direction='out')
    #axes[0].set_xticklabels(["Constant", "Growth"], minor=False, fontsize=6)


    #axes[0].set_yticklabels(rowLabels, minor=False, fontsize=6)
    #for y in range(data.shape[0]):
    #    for x in range(data.shape[1]):
    #        val = data[y, x]
    #        val *= 100
    #        if val > 50:
    #            c = '0.9'
    #        else:
    #            c = 'black'
    #        axes[0].text(x + 0.5, y + 0.5, '%.1f%%' % val, horizontalalignment='center', verticalalignment='center', color=c, fontsize=6)


    #axes[0].set_title(results["name"]+"\n"+labels,fontsize=6)

    #results = pickle.load(open(resultsFile2 , "rb"))

    #predictions = results["predictions"]
    #realValues = results["Y_test"]

    #const, expan = [], []

    #const_const, const_expan, expan_const, expan_expan = 0,0,0,0
    #const_total, expan_total = 0,0
    #for i, val in enumerate(realValues):
    #    if val[0] == 1.0:
    #        const_total+=1
    #        const.append(1.0 - predictions[i][0])
    #        if predictions[i][0] > 0.5:
    #            const_const+=1
    #        if predictions[i][1] > 0.5:
    #            const_expan+=1
    #    else:
    #        expan_total+=1
    #        expan.append(predictions[i][1])
    #        if predictions[i][0] > 0.5:
    #            expan_const+=1
    #        if predictions[i][1] > 0.5:
    #            expan_expan+=1
    #np.array(const)
    #np.array(expan)

    #ce = round(cross_entropy(predictions,realValues),4)
    #labels = "Cross entropy = " + str(ce)

    #data=np.array([[const_const/float(const_total),const_expan/float(const_total)],[expan_const/float(expan_total),expan_expan/float(expan_total)]])
    #rowLabels = ["Constant", "Growth"]
    #heatmap = axes[1].pcolor(data, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    #cbar = plt.colorbar(heatmap, cmap=plt.cm.Blues, ax=axes[1])
    #cbar.set_label('Proportion assigned to class', rotation=270, labelpad=20)

    ## put the major ticks at the middle of each cell
    #axes[1].set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    #axes[1].set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    #axes[1].invert_yaxis()
    #axes[1].xaxis.tick_top()

    #plt.tick_params(axis='y', which='both', right='off')
    #plt.tick_params(axis='x', which='both', direction='out')
    #axes[1].set_xticklabels(["Constant", "Growth"], minor=False, fontsize=6)


    #axes[1].set_yticklabels(rowLabels, minor=False, fontsize=6)
    #for y in range(data.shape[0]):
    #    for x in range(data.shape[1]):
    #        val = data[y, x]
    #        val *= 100
    #        if val > 50:
    #            c = '0.9'
    #        else:
    #            c = 'black'
    #        axes[1].text(x + 0.5, y + 0.5, '%.1f%%' % val, horizontalalignment='center', verticalalignment='center', color=c, fontsize=6)


    #axes[1].set_title(results["name"]+"\n"+labels,fontsize=6)

    #fig.subplots_adjust(left=.15, bottom=.05, right=.85, top=0.87, hspace = 0.6, wspace=0.4)
    #height = 4.00
    #width = 4.00

    #fig.set_size_inches(height, width)
    #fig.savefig(saveas)

#-------------------------------------------------------------------------------------------

def getMeanSDMax(trainDir):
    '''
    get the mean and standard deviation of rho from training set

    str -> int,int,int

    '''
    info = pickle.load(open(trainDir+"/info.p","rb"))
    rho = info["rho"]
    segSites = info["segSites"]
    tar_mean = np.mean(rho,axis=0)
    tar_sd = np.std(rho,axis=0)
    return tar_mean,tar_sd,max(segSites)

#-------------------------------------------------------------------------------------------

def unNormalize(mean,sd,data):
    '''
    un-zcore-ify. do the inverse to get real value predictions

    float,float,float[] -> float[]
    '''

    data *= sd
    data += mean  ##comment this line out for GRU_TUNED84_RELU
    return data

#-------------------------------------------------------------------------------------------

def plotParametricBootstrap(results,saveas):

    '''
    Use the location of "out" paramerter to parametric bootstrap
    as input to plot the results of said para-boot
    '''

    stats = pickle.load(open(results,'rb'))
    x = stats["rho"]

    fig, ax = plt.subplots()


    for i,s in enumerate(stats):
        if(i == 0):
            continue

        ax.plot(x,stats[s])

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    fig.savefig(saveas)

    return None

