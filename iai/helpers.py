'''
Authors: Jared Galloway, Jeff Adrion
'''

from iai.imports import *
from iai.sequenceBatchGenerator import *

#-------------------------------------------------------------------------------------------

def log_prior(theta):
    ''' The natural logarithm of the prior probability. '''

    lp = 0.

    # unpack the model parameters from the tuple
    m, c = theta

    # uniform prior on c
    cmin = -10. # lower range of prior
    cmax = 10.  # upper range of prior

    # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
    lp = 0. if cmin < c < cmax else -np.inf

    # Gaussian prior on m
    mmu = 3.     # mean of the Gaussian prior
    msigma = 10. # standard deviation of the Gaussian prior
    lp -= 0.5*((m - mmu)/msigma)**2

    return lp

#-------------------------------------------------------------------------------------------

def log_like(theta, data, sigma, x):
    '''The natural logarithm of the likelihood.'''

    # unpack the model parameters
    m, c = theta

    # evaluate the model
    md = straight_line(x, m, c)

    # return the log likelihood
    return -0.5 * np.sum(((md - data)/sigma)**2)

#-------------------------------------------------------------------------------------------

def log_post(theta, data, sigma, x):
    '''The natural logarithm of the posterior.'''

    return logprior(theta) + loglike(theta, data, sigma, x)

#-------------------------------------------------------------------------------------------

def log_prob(x, mu, icov):
    return -0.5 * np.dot(np.dot((x-mu).T,icov),(x-mu))

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

def runModels_cleverhans_tf2(ModelFuncPointer,
            ModelName,
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
            attackFraction=0.0,
            attackBatchSize=None,
            rep=None,
            FGSM=False,
            PGD=False):


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

    # Store original batch_size for train and vali sets and total numReps
    og_train_bs = TrainParams["batchSize"]
    tmpDir = TrainParams['treesDirectory']
    infoP = pickle.load(open(os.path.join(tmpDir,"info.p"),"rb"))
    og_train_numReps = infoP["numReps"]

    og_vali_bs = ValiParams["batchSize"]
    tmpDir = ValiParams['treesDirectory']
    infoP = pickle.load(open(os.path.join(tmpDir,"info.p"),"rb"))
    og_vali_numReps = infoP["numReps"]

    og_test_bs = TestParams["batchSize"]
    og_test_dir = TestParams['treesDirectory']
    infoP = pickle.load(open(os.path.join(og_test_dir,"info.p"),"rb"))
    og_test_numReps = infoP["numReps"]

    # Call the generator
    x,y = TrainGenerator.__getitem__(0)

    # If TestGenerator is called after model.fit the random shuffling is not the same, even with same seed
    x_test,y_test = TestGenerator.__getitem__(0)
    img_rows, img_cols = x_test.shape[1], x_test.shape[2]

    ## define model
    model = ModelFuncPointer(x,y)
    # Early stopping and saving the best weights
    callbacks_list = [
            EarlyStopping(
                monitor='val_loss',
                verbose=1,
                min_delta=0.01,
                patience=25),
            ModelCheckpoint(
                filepath=network[1],
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
    else:
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

    # Metrics to track the different accuracies.
    test_acc_clean = tf.metrics.CategoricalAccuracy()
    test_acc_fgsm = tf.metrics.CategoricalAccuracy()
    test_acc_pgd = tf.metrics.CategoricalAccuracy()

    # predict on clean test examples
    print("\nPredicting on clean examples...")
    y_pred = model.predict(x_test)
    test_acc_clean(y_test, y_pred)
    print('test acc on clean examples (%): {:.3f}'.format(test_acc_clean.result() * 100))

    if FGSM:
        # predict on adversarial test examples using FGSM
        print("\nAttacking using Fast Gradient Sign Method...")
        fgsm_params = {'eps': 1.0,
                'norm': np.inf,
                'clip_min': 0.0,
                'clip_max': 1.0}

        # define the attack generator for test examples
        adv_test_params = copy.deepcopy(TestParams)
        adv_test_params["model"] = model
        adv_test_params["attackName"] = "fgsm"
        adv_test_params["attackParams"] = fgsm_params
        adv_test_params["attackFraction"] = 1.0
        adv_test_params["writeAttacks"] = True
        adv_test_params["batchSize"] = attackBatchSize
        attackGen_test = SequenceBatchGenerator(**adv_test_params)
        # attack the entire test set and write adversarial examples to disk
        print("Attacking the test set in batches of %s..."%(attackBatchSize))
        num_batches = int(np.ceil(og_test_numReps/float(adv_test_params["batchSize"])))
        t0 = time.perf_counter()
        for i in range(num_batches):
            attackGen_test.__getitem__(i)
            progress_bar(i/float(num_batches))
        progress_bar(num_batches/float(num_batches))
        t1 = time.perf_counter()
        print("\nAverage time per FGSM attack (s):", round((t1-t0)/float(og_test_numReps),6))

        # reset generator parameters
        adv_test_dir = og_test_dir + "_fgsm_rep%s"%(rep)
        cmd = "cp %s %s"%(os.path.join(og_test_dir,"info.p"), os.path.join(adv_test_dir,"info.p"))
        os.system(cmd)
        adv_test_params["treesDirectory"] = adv_test_dir
        adv_test_params["attackFraction"] = 0.0
        adv_test_params["writeAttacks"] = False
        adv_test_params["batchSize"] = og_test_bs
        attackGen_test = SequenceBatchGenerator(**adv_test_params)

        x_fgsm, y_fgsm = attackGen_test.__getitem__(0)

        print("\nPredicting on FGSM examples...")
        y_pred_fgsm = model.predict(x_fgsm)
        test_acc_fgsm(y_test, y_pred_fgsm)
        print('test acc on FGSM adversarial examples (%): {:.3f}'.format(test_acc_fgsm.result() * 100))

    if PGD:
        # predict on adversarial test examples using PGD
        print("\nAttacking using Projected Gradient Descent...")
        pgd_params = {'eps': 1.0,
                'eps_iter': 1.0,
                'nb_iter': 40,
                'norm': np.inf,
                'clip_min': 0.0,
                'clip_max': 1.0,
                'sanity_checks': False}

        # define the attack generator for test examples
        adv_test_params = copy.deepcopy(TestParams)
        adv_test_params["model"] = model
        adv_test_params["attackName"] = "pgd"
        adv_test_params["attackParams"] = pgd_params
        adv_test_params["attackFraction"] = 1.0
        adv_test_params["writeAttacks"] = True
        adv_test_params["batchSize"] = attackBatchSize
        attackGen_test = SequenceBatchGenerator(**adv_test_params)

        # attack the entire test set and write adversarial examples to disk
        print("Attacking the test set in batches of %s..."%(attackBatchSize))
        num_batches = int(np.ceil(og_test_numReps/float(adv_test_params["batchSize"])))
        t0 = time.perf_counter()
        for i in range(num_batches):
            attackGen_test.__getitem__(i)
            progress_bar(i/float(num_batches))
        progress_bar(num_batches/float(num_batches))
        t1 = time.perf_counter()
        print("\nAverage time per PGD attack (s):", round((t1-t0)/float(og_test_numReps),6))

        # reset generator parameters
        adv_test_dir = og_test_dir + "_pgd_rep%s"%(rep)
        cmd = "cp %s %s"%(os.path.join(og_test_dir,"info.p"), os.path.join(adv_test_dir,"info.p"))
        os.system(cmd)
        adv_test_params["treesDirectory"] = adv_test_dir
        adv_test_params["attackFraction"] = 0.0
        adv_test_params["writeAttacks"] = False
        adv_test_params["batchSize"] = og_test_bs
        attackGen_test = SequenceBatchGenerator(**adv_test_params)

        x_pgd, y_pgd = attackGen_test.__getitem__(0)

        print("\nPredicting on PGD examples...")
        y_pred_pgd = model.predict(x_pgd)
        test_acc_pgd(y_test, y_pred_pgd)
        print('test acc on PGD adversarial examples (%): {:.3f}'.format(test_acc_pgd.result() * 100))

    # Tally results
    print("results written to: ",resultsFile)
    history.history['loss'] = np.array(history.history['loss'])
    history.history['val_loss'] = np.array(history.history['val_loss'])
    history.history['predictions'] = np.array(y_pred)
    if FGSM:
        history.history['predictions_fgsm'] = np.array(y_pred_fgsm)
    if PGD:
        history.history['predictions_pgd'] = np.array(y_pred_pgd)
    history.history['Y_test'] = np.array(y_test)
    history.history['name'] = ModelName
    pickle.dump(history.history, open(resultsFile, "wb" ))

    if FGSM or PGD:
        # Save genotype images for testset
        print("\nSaving adversarial images...")
        if rep:
            imageDir = os.path.join(ProjectDir,"test_images"+"_rep%s"%(rep))
        else:
            imageDir = os.path.join(ProjectDir,"test_images")
        if not os.path.exists(imageDir):
            os.makedirs(imageDir)
        for i in range(x_test.shape[0]):
            clean_gmFILE = os.path.join(imageDir,"examp{}_clean.npy".format(i))
            clean_image = x_test[i]
            clean_imageFILE = os.path.join(imageDir,"examp{}_clean.png".format(i))
            plt.imsave(clean_imageFILE, clean_image)
            if FGSM:
                fgsm_gmFILE = os.path.join(imageDir,"examp{}_fgsm.npy".format(i))
                fgsm_imageFILE = os.path.join(imageDir,"examp{}_fgsm.png".format(i))
                fgsm_delta_imageFILE = os.path.join(imageDir,"examp{}_fgsm_delta.png".format(i))
                fgsm_image = x_fgsm[i]
                fgsm_delta_image = clean_image - fgsm_image
                plt.imsave(fgsm_imageFILE, fgsm_image)
                plt.imsave(fgsm_delta_imageFILE, fgsm_delta_image)
            if PGD:
                pdg_gmFILE = os.path.join(imageDir,"examp{}_pgd.npy".format(i))
                pgd_imageFILE = os.path.join(imageDir,"examp{}_pgd.png".format(i))
                pgd_delta_imageFILE = os.path.join(imageDir,"examp{}_pgd_delta.png".format(i))
                pgd_image = x_pgd[i]
                pgd_delta_image = clean_image - pgd_image
                plt.imsave(pgd_imageFILE, pgd_image)
                plt.imsave(pgd_delta_imageFILE, pgd_delta_image)
            progress_bar(i/float(x_test.shape[0]))
        progress_bar(x_test.shape[0]/float(x_test.shape[0]))
        print("\n")

    if FGSM:
        ########## Adversarial training (FGSM) #############
        ## similar objects as above except these have the extension _fgsm and _pgd
        print("Repeating the process, training training on adversarial examples (FGSM)")
        # define the attack generator for training examples
        adv_train_params = copy.deepcopy(TrainParams)
        adv_train_params["model"] = model
        adv_train_params["attackName"] = "fgsm"
        adv_train_params["attackParams"] = fgsm_params
        adv_train_params["attackFraction"] = 1.0
        adv_train_params["writeAttacks"] = True
        adv_train_params["batchSize"] = attackBatchSize
        attackGen_train = SequenceBatchGenerator(**adv_train_params)

        # attack the entire training set and write adversarial examples to disk
        print("Attacking the training set in batches of %s..."%(attackBatchSize))
        num_batches = int(np.ceil(og_train_numReps/float(adv_train_params["batchSize"])))
        for i in range(num_batches):
            x_train,y_train = attackGen_train.__getitem__(i)
            progress_bar(i/float(num_batches))
        progress_bar(num_batches/float(num_batches))

        # define the attack generator for validation examples
        adv_vali_params = copy.deepcopy(ValiParams)
        adv_vali_params["model"] = model
        adv_vali_params["attackName"] = "fgsm"
        adv_vali_params["attackParams"] = fgsm_params
        adv_vali_params["attackFraction"] = 1.0
        adv_vali_params["writeAttacks"] = True
        adv_vali_params["batchSize"] = attackBatchSize
        attackGen_vali = SequenceBatchGenerator(**adv_vali_params)

        # attack the entire validation set and write adversarial examples to disk
        print("\nAttacking the validation set in batches of %s..."%(attackBatchSize))
        num_batches = int(np.ceil(og_vali_numReps/float(adv_vali_params["batchSize"])))
        for i in range(num_batches):
            x_vali,y_vali = attackGen_vali.__getitem__(i)
            progress_bar(i/float(num_batches))
        progress_bar(num_batches/float(num_batches))

        # reset generator parameters in preperation for model fit
        adv_train_params["attackFraction"] = attackFraction
        adv_train_params["writeAttacks"] = False
        adv_train_params["batchSize"] = og_train_bs
        attackGen_train = SequenceBatchGenerator(**adv_train_params)
        adv_vali_params["attackFraction"] = attackFraction
        adv_vali_params["writeAttacks"] = False
        adv_vali_params["batchSize"] = og_vali_bs
        attackGen_vali = SequenceBatchGenerator(**adv_vali_params)

        ## define the new model
        print('\n')
        model_fgsm = ModelFuncPointer(x_train,y_train)

        ## Early stopping and saving the best weights
        callbacks_list_fgsm = [
                EarlyStopping(
                    monitor='val_loss',
                    verbose=1,
                    min_delta=0.01,
                    patience=25),
                ModelCheckpoint(
                    filepath=network[1].replace(".h5","_fgsm.h5"),
                    monitor='val_loss',
                    save_best_only=True)
                ]

        # Train the network
        history_fgsm = model_fgsm.fit(x=attackGen_train,
            steps_per_epoch=epochSteps,
            epochs=numEpochs,
            validation_data=attackGen_vali,
            callbacks=callbacks_list_fgsm,
            use_multiprocessing=False,
            verbose=2)

        # Write the network
        if(network != None):
            ##serialize model_fgsm to JSON
            model_json_fgsm = model_fgsm.to_json()
            with open(network[0].replace(".json","_fgsm.json"), "w") as json_file:
                json_file.write(model_json_fgsm)

        # Load json and create model
        if(network != None):
            jsonFILE = open(network[0].replace(".json","_fgsm.json"),"r")
            loadedModel_fgsm = jsonFILE.read()
            jsonFILE.close()
            model_fgsm=model_from_json(loadedModel_fgsm)
            model_fgsm.load_weights(network[1].replace(".h5","_fgsm.h5"))
        else:
            print("Error: model_fgsm and weights_fgsm not loaded")
            sys.exit(1)

        # Metrics to track the different accuracies.
        test_acc_clean_fgsm = tf.metrics.CategoricalAccuracy()
        test_acc_fgsm_fgsm = tf.metrics.CategoricalAccuracy()
        test_acc_pgd_fgsm = tf.metrics.CategoricalAccuracy()

        # predict on clean test examples
        print("Predicting on clean examples...")
        y_pred_fgsm = model_fgsm.predict(x_test)
        test_acc_clean_fgsm(y_test, y_pred_fgsm)
        print('test acc on clean examples (%): {:.3f}'.format(test_acc_clean_fgsm.result() * 100))

        # predict on adversarial test examples using FGSM
        print("Predicting on FGSM examples...")
        y_pred_fgsm_fgsm = model_fgsm.predict(x_fgsm)
        test_acc_fgsm_fgsm(y_test, y_pred_fgsm_fgsm)
        print('test acc on FGSM adversarial examples (%): {:.3f}'.format(test_acc_fgsm_fgsm.result() * 100))

        if PGD:
            # predict on adversarial test examples using PGD
            print("Predicting on PGD examples...")
            y_pred_pgd_fgsm = model_fgsm.predict(x_pgd)
            test_acc_pgd_fgsm(y_test, y_pred_pgd_fgsm)
            print('test acc on PGD adversarial examples (%): {:.3f}'.format(test_acc_pgd_fgsm.result() * 100))

        ## write results
        print("results_fgsm written to: ",resultsFile.replace(".p","_fgsm.p"))
        history_fgsm.history['loss'] = np.array(history_fgsm.history['loss'])
        history_fgsm.history['val_loss'] = np.array(history_fgsm.history['val_loss'])
        history_fgsm.history['predictions'] = np.array(y_pred_fgsm)
        history_fgsm.history['predictions_fgsm'] = np.array(y_pred_fgsm_fgsm)
        if PGD:
            history_fgsm.history['predictions_pgd'] = np.array(y_pred_pgd_fgsm)
        history_fgsm.history['Y_test'] = np.array(y_test)
        history_fgsm.history['name'] = ModelName
        pickle.dump(history_fgsm.history, open( resultsFile.replace(".p","_fgsm.p"), "wb" ))


    if PGD:
        ########## Adversarial training (PGD) #############
        ## similar objects as above except these have the extension _fgsm and _pgd
        print("\nRepeating the process, training training on adversarial examples (PGD)")
        # define the attack generator for training examples
        adv_train_params = copy.deepcopy(TrainParams)
        adv_train_params["model"] = model
        adv_train_params["attackName"] = "pgd"
        adv_train_params["attackParams"] = pgd_params
        adv_train_params["attackFraction"] = 1.0
        adv_train_params["writeAttacks"] = True
        adv_train_params["batchSize"] = attackBatchSize
        attackGen_train = SequenceBatchGenerator(**adv_train_params)

        # attack the entire training set and write adversarial examples to disk
        print("Attacking the training set in batches of %s..."%(attackBatchSize))
        num_batches = int(np.ceil(og_train_numReps/float(adv_train_params["batchSize"])))
        for i in range(num_batches):
            x_train,y_train = attackGen_train.__getitem__(i)
            progress_bar(i/float(num_batches))

        # define the attack generator for validation examples
        adv_vali_params = copy.deepcopy(ValiParams)
        adv_vali_params["model"] = model
        adv_vali_params["attackName"] = "pgd"
        adv_vali_params["attackParams"] = pgd_params
        adv_vali_params["attackFraction"] = 1.0
        adv_vali_params["writeAttacks"] = True
        adv_vali_params["batchSize"] = attackBatchSize
        attackGen_vali = SequenceBatchGenerator(**adv_vali_params)

        # attack the entire validation set and write adversarial examples to disk
        print("\nAttacking the validation set in batches of %s..."%(attackBatchSize))
        num_batches = int(np.ceil(og_vali_numReps/float(adv_vali_params["batchSize"])))
        for i in range(num_batches):
            x_vali,y_vali = attackGen_vali.__getitem__(i)
            progress_bar(i/float(num_batches))

        # reset generator parameters in preperation for model fit
        adv_train_params["attackFraction"] = attackFraction
        adv_train_params["writeAttacks"] = False
        adv_train_params["batchSize"] = og_train_bs
        attackGen_train = SequenceBatchGenerator(**adv_train_params)
        adv_vali_params["attackFraction"] = attackFraction
        adv_vali_params["writeAttacks"] = False
        adv_vali_params["batchSize"] = og_vali_bs
        attackGen_vali = SequenceBatchGenerator(**adv_vali_params)

        ## define the new model
        print('\n')
        model_pgd = ModelFuncPointer(x_train,y_train)

        ## Early stopping and saving the best weights
        callbacks_list_pgd = [
                EarlyStopping(
                    monitor='val_loss',
                    verbose=1,
                    min_delta=0.01,
                    patience=25),
                ModelCheckpoint(
                    filepath=network[1].replace(".h5","_pgd.h5"),
                    monitor='val_loss',
                    save_best_only=True)
                ]

        # Train the network
        history_pgd = model_pgd.fit(x=attackGen_train,
            steps_per_epoch= epochSteps,
            epochs=numEpochs,
            validation_data=attackGen_vali,
            callbacks=callbacks_list_pgd,
            use_multiprocessing=False,
            verbose=2)

        # Write the network
        if(network != None):
            ##serialize model_pgd to JSON
            model_json_pgd = model_pgd.to_json()
            with open(network[0].replace(".json","_pgd.json"), "w") as json_file:
                json_file.write(model_json_pgd)

        # Load json and create model
        if(network != None):
            jsonFILE = open(network[0].replace(".json","_pgd.json"),"r")
            loadedModel_pgd = jsonFILE.read()
            jsonFILE.close()
            model_pgd=model_from_json(loadedModel_pgd)
            model_pgd.load_weights(network[1].replace(".h5","_pgd.h5"))
        else:
            print("Error: model_pgd and weights_pgd not loaded")
            sys.exit(1)

        # Metrics to track the different accuracies.
        test_acc_clean_pgd = tf.metrics.CategoricalAccuracy()
        test_acc_fgsm_pgd = tf.metrics.CategoricalAccuracy()
        test_acc_pgd_pgd = tf.metrics.CategoricalAccuracy()

        # predict on clean test examples
        print("Predicting on clean examples...")
        y_pred_pgd = model_pgd.predict(x_test)
        test_acc_clean_pgd(y_test, y_pred_pgd)
        print('test acc on clean examples (%): {:.3f}'.format(test_acc_clean_pgd.result() * 100))

        if FGSM:
            # predict on adversarial test examples using FGSM
            print("Predicting on FGSM examples...")
            y_pred_fgsm_pgd = model_pgd.predict(x_fgsm)
            test_acc_fgsm_pgd(y_test, y_pred_fgsm_pgd)
            print('test acc on FGSM adversarial examples (%): {:.3f}'.format(test_acc_fgsm_pgd.result() * 100))

        # predict on adversarial test examples using PGD
        print("Predicting on PGD examples...")
        y_pred_pgd_pgd = model_pgd.predict(x_pgd)
        test_acc_pgd_pgd(y_test, y_pred_pgd_pgd)
        print('test acc on PGD adversarial examples (%): {:.3f}'.format(test_acc_pgd_pgd.result() * 100))

        ## write results
        print("results_pgd written to: ",resultsFile.replace(".p","_pgd.p"))
        history_pgd.history['loss'] = np.array(history_pgd.history['loss'])
        history_pgd.history['val_loss'] = np.array(history_pgd.history['val_loss'])
        history_pgd.history['predictions'] = np.array(y_pred_pgd)
        if FGSM:
            history_pgd.history['predictions_fgsm'] = np.array(y_pred_fgsm_pgd)
        history_pgd.history['predictions_pgd'] = np.array(y_pred_pgd_pgd)
        history_pgd.history['Y_test'] = np.array(y_test)
        history_pgd.history['name'] = ModelName
        pickle.dump(history_pgd.history, open( resultsFile.replace(".p","_pgd.p"), "wb" ))

    ######### write log ###########
    outLog = resultsFile.replace(".p","_log.txt")
    with open(outLog, "w") as fOUT:
        fOUT.write("Before adversarial training\n")
        fOUT.write("===========================\n")
        fOUT.write('test acc on clean examples (%): {:.3f}\n'.format(test_acc_clean.result() * 100))
        if FGSM:
            fOUT.write('test acc on FGSM adversarial examples (%): {:.3f}\n'.format(test_acc_fgsm.result() * 100))
        if PGD:
            fOUT.write('test acc on PGD adversarial examples (%): {:.3f}\n'.format(test_acc_pgd.result() * 100))
        if FGSM:
            fOUT.write("After adversarial training (fgsm attack)\n")
            fOUT.write("===========================\n")
            fOUT.write('test acc on clean examples (%): {:.3f}\n'.format(test_acc_clean_fgsm.result() * 100))
            fOUT.write('test acc on FGSM adversarial examples (%): {:.3f}\n'.format(test_acc_fgsm_fgsm.result() * 100))
            if PGD:
                fOUT.write('test acc on PGD adversarial examples (%): {:.3f}\n'.format(test_acc_pgd_fgsm.result() * 100))
        if PGD:
            fOUT.write("After adversarial training (pgd attack)\n")
            fOUT.write("===========================\n")
            fOUT.write('test acc on clean examples (%): {:.3f}\n'.format(test_acc_clean_pgd.result() * 100))
            if FGSM:
                fOUT.write('test acc on FGSM adversarial examples (%): {:.3f}\n'.format(test_acc_fgsm_pgd.result() * 100))
            fOUT.write('test acc on PGD adversarial examples (%): {:.3f}\n'.format(test_acc_pgd_pgd.result() * 100))

    return

#-------------------------------------------------------------------------------------------

def predict_cleverhans_tf2(ModelFuncPointer,
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
            FGSM=False,
            PGD=False,
            task=None):

    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuID)

    ## The following code block appears necessary for running with tf2 and cudnn
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import Session
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    Session(config=config)
    ###

    ########### Prediction on non-adversarial trained network #############
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

    # Metrics to track the different accuracies.
    test_acc_clean = tf.metrics.CategoricalAccuracy()
    test_acc_fgsm = tf.metrics.CategoricalAccuracy()
    test_acc_pgd = tf.metrics.CategoricalAccuracy()

    # Read all clean test data into memory
    x_test, y_test = TestGenerator.__getitem__(0)
    predictions = model.predict(x_test)

    #replace predictions and Y_test in results file
    history= pickle.load(open(resultsFile, "rb"))
    tmp = []
    if task == "expansion":
        task_key = "gr"
    if task == "admixture":
        task_key = "m"
    for gr in test_info[task_key]:
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


    if FGSM:
        ########### Prediction on adversarial trained network (FGSM) #############
        # Load json and create model
        if(network != None):
            jsonFILE = open(network[0].replace(".json","_fgsm.json"),"r")
            loadedModel_fgsm = jsonFILE.read()
            jsonFILE.close()
            model_fgsm=model_from_json(loadedModel_fgsm)
            model_fgsm.load_weights(network[1].replace(".h5","_fgsm.h5"))
        else:
            print("Error: model_fgsm and weights_fgsm not loaded")
            sys.exit(1)

        predictions_fgsm = model_fgsm.predict(x_test)

        #replace predictions and T_test in results file
        history_fgsm = pickle.load(open(resultsFile.replace(".p","_fgsm.p"), "rb"))
        tmp = []
        for gr in test_info[task_key]:
            if gr > 0.0:
                tmp.append([0.0,1.0])
            else:
                tmp.append([1.0,0.0])
        history_fgsm["Y_test"] = np.array(tmp)
        history_fgsm['predictions'] = np.array(predictions_fgsm)
        test_acc_fgsm(y_test, predictions_fgsm)

        # rewrite new results file
        newResultsFile = resultsFile.replace(".p","_fgsm_params%s.p"%(paramsID))
        print("new results written to: ", newResultsFile)
        pickle.dump(history_fgsm, open(newResultsFile, "wb"))


    if PGD:
        ########### Prediction on adversarial trained network (PGD) #############
        # Load json and create model
        if(network != None):
            jsonFILE = open(network[0].replace(".json","_pgd.json"),"r")
            loadedModel_pgd = jsonFILE.read()
            jsonFILE.close()
            model_pgd=model_from_json(loadedModel_pgd)
            model_pgd.load_weights(network[1].replace(".h5","_pgd.h5"))
        else:
            print("Error: model_pgd and weights_pgd not loaded")
            sys.exit(1)

        predictions_pgd = model_pgd.predict(x_test)

        #replace predictions and T_test in results file
        history_pgd = pickle.load(open(resultsFile.replace(".p","_pgd.p"), "rb"))
        tmp = []
        for gr in test_info[task_key]:
            if gr > 0.0:
                tmp.append([0.0,1.0])
            else:
                tmp.append([1.0,0.0])
        history_pgd["Y_test"] = np.array(tmp)
        history_pgd['predictions'] = np.array(predictions_pgd)
        test_acc_pgd(y_test, predictions_pgd)

        ## print results
        print('test acc on clean examples (%): {:.3f}'.format(test_acc_clean.result() * 100))
        print('test acc on FGSM adversarial examples (%): {:.3f}'.format(test_acc_fgsm.result() * 100))
        print('test acc on PGD adversarial examples (%): {:.3f}'.format(test_acc_pgd.result() * 100))

        # rewrite new results file
        newResultsFile = resultsFile.replace(".p","_pgd_params%s.p"%(paramsID))
        print("new results written to: ", newResultsFile)
        pickle.dump(history_pgd, open(newResultsFile, "wb"))

    ######### write log ###########
    outLog = resultsFile.replace(".p","_log_params%s.txt"%(paramsID))
    with open(outLog, "w") as fOUT:
        fOUT.write("Before adversarial training\n")
        fOUT.write("===========================\n")
        fOUT.write('test acc on test_paramsB examples (%): {:.3f}\n'.format(test_acc_clean.result() * 100))
        if FGSM:
            fOUT.write("After adversarial training (fgsm attack)\n")
            fOUT.write("===========================\n")
            fOUT.write('test acc on test_paramsB examples (%): {:.3f}\n'.format(test_acc_fgsm.result() * 100))
        if PGD:
            fOUT.write("After adversarial training (pgd attack)\n")
            fOUT.write("===========================\n")
            fOUT.write('test acc on test_paramsB examples (%): {:.3f}\n'.format(test_acc_pgd.result() * 100))

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

def sort_min_diff(amat):
    '''this function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted by genetic similarity.
    this problem is NP, so here we use a nearest neighbors approx.  it's not perfect, but it's fast and generally performs ok.
    assumes your input matrix is a numpy array'''

    mb = NearestNeighbors(len(amat), metric='manhattan').fit(amat)
    v = mb.kneighbors(amat)
    smallest = np.argmin(v[0].sum(axis=1))
    return amat[v[1][smallest]]

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

def plotResultsSoftmax2Heatmap(resultsFile,saveas,admixture):

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
    if admixture:
        rowLabels = ["No admixture", "Admixture"]
    else:
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
    if admixture:
        axes[0].set_xticklabels(["No admixture", "Admixture"], minor=False, fontsize=6)
    else:
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

def plotResultsSoftmax2HeatmapMis(resultsFile, resultsFile2, saveas, admixture):

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
    if admixture:
        rowLabels = ["No admixture", "Admixture"]
    else:
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
    if admixture:
        axes[0].set_xticklabels(["No admixture", "Admixture"], minor=False, fontsize=6)
    else:
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
    if admixture:
        rowLabels = ["No admixture", "Admixture"]
    else:
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
    if admixture:
        axes[1].set_xticklabels(["No admixture", "Admixture"], minor=False, fontsize=6)
    else:
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

#-------------------------------------------------------------------------------------------

def snakemake_stitch_info(projectDir, seed=None, reps=None):
    '''
    combine the info files
    created using snakemake into a directory structure
    that looks as if it was created using the standard
    iai-simulate pipeline
    '''

    ## Make directories if they do not exist
    trainDir = os.path.join(projectDir,"train")
    valiDir = os.path.join(projectDir,"vali")
    testDir = os.path.join(projectDir,"test")
    networkDir = os.path.join(projectDir,"networks")

    ## Might need to add some info keys if using grid params?
    info_keys = ["rho","mu","m","segSites","seed","gr","ne"]

    ## Combine the `info.p` files
    minSegSites = float("inf")
    for i,new_dir in enumerate([trainDir,valiDir,testDir]):
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        new_info_file = {}
        for j in range(reps):
            trainRep = os.path.join(projectDir,"rep{}".format(j+1),"train")
            valiRep = os.path.join(projectDir,"rep{}".format(j+1),"vali")
            testRep = os.path.join(projectDir,"rep{}".format(j+1),"test")
            networkRep = os.path.join(projectDir,"rep{}".format(j+1),"networks")
            rep_dirs = [trainRep,valiRep,testRep]
            info_file = pickle.load(open(os.path.join(rep_dirs[i],"info.p"),"rb"))
            try:
                new_info_file["numReps"] += info_file["numReps"]
                for key in info_keys:
                    new_array = np.concatenate((new_info_file[key], info_file[key]), axis=None)
                    new_info_file[key] = new_array
            except KeyError:
                new_info_file = info_file
        S_min = min(new_info_file["segSites"])
        minSegSites = min(minSegSites, S_min)
        pickle.dump(new_info_file, open(os.path.join(new_dir, "info.p"), "wb"))

    ## Add the `simPars.p` file
    if not os.path.exists(networkDir):
        os.makedirs(networkDir)
    simPars = pickle.load(open(os.path.join(networkRep,"simPars.p"),"rb"))
    simPars["seed"] = seed
    if os.path.basename(projectDir):
        simPars["bn"] = os.path.basename(projectDir)
    else:
        simPars["bn"] = projectDir.split("/")[-2]
    simPars["minSegSites"] = minSegSites
    pickle.dump(simPars, open(os.path.join(networkDir,"simPars.p"),"wb"))

    return None

#-------------------------------------------------------------------------------------------

def snakemake_stitch_sims(projectDir, rep_dir, idx, nTrain, nVali, nTest, trim=False):
    '''
    combine the simulation files
    created using snakemake into a directory structure
    that looks as if it was created using the standard
    iai-simulate pipeline
    '''

    ## Move and rename the simulation files
    trainDir = os.path.join(projectDir,"train")
    valiDir = os.path.join(projectDir,"vali")
    testDir = os.path.join(projectDir,"test")
    networkDir = os.path.join(projectDir,"networks")
    minSegSites = pickle.load(open(os.path.join(networkDir,"simPars.p"),"rb"))["minSegSites"]
    sims_per_rep = [nTrain, nVali, nTest]
    for i,new_dir in enumerate([trainDir,valiDir,testDir]):
        if trim:
            print("\nTrimming genotype and position .npy files in %s to %s SNPs"%(new_dir,minSegSites))
        new_index = (int(idx)-1) * sims_per_rep[i]
        trainRep = os.path.join(rep_dir,"train")
        valiRep = os.path.join(rep_dir,"vali")
        testRep = os.path.join(rep_dir,"test")
        rep_dirs = [trainRep,valiRep,testRep]
        for j in range(sims_per_rep[i]):
            H_orig_file = os.path.join(rep_dirs[i], "{}_haps.npy".format(j))
            P_orig_file = os.path.join(rep_dirs[i], "{}_pos.npy".format(j))
            H_new_file = os.path.join(new_dir, "{}_haps.npy".format(new_index))
            P_new_file = os.path.join(new_dir, "{}_pos.npy".format(new_index))
            H = np.load(H_orig_file)
            P = np.load(P_orig_file)
            if trim:
                H = H[:minSegSites]
                P = P[:minSegSites]
            np.save(H_new_file,H)
            np.save(P_new_file,P)
            new_index += 1

    done_file = os.path.join(rep_dir,"done.txt")
    with open(done_file, "w") as fIN:
        fIN.write("done")

    # for storage efficiency, remove files only after trim is complete
    for i,new_dir in enumerate([trainDir,valiDir,testDir]):
        trainRep = os.path.join(rep_dir,"train")
        valiRep = os.path.join(rep_dir,"vali")
        testRep = os.path.join(rep_dir,"test")
        rep_dirs = [trainRep,valiRep,testRep]
        for j in range(sims_per_rep[i]):
            H_orig_file = os.path.join(rep_dirs[i], "{}_haps.npy".format(j))
            P_orig_file = os.path.join(rep_dirs[i], "{}_pos.npy".format(j))
            os.remove(H_orig_file)
            os.remove(P_orig_file)

    return None

#-------------------------------------------------------------------------------------------

def snakemake_remove_rep_dirs(projectDir, reps):
    '''
    remove all the replicate directory structure
    '''

    for j in range(reps):
        rep_dir = os.path.join(projectDir,"rep{}".format(j+1))
        shutil.rmtree(rep_dir)

    done_file = os.path.join(projectDir,"done.txt")
    with open(done_file, "w") as fIN:
        fIN.write("done")

    print("Snakefile done")

    return None

#-------------------------------------------------------------------------------------------


def worker_downsample(task_q, result_q, params):
    while True:
        try:
            mpID, nth_job = task_q.get()
            subsample_indices, minSites, from_dir, to_dir = params
            for i in mpID:
                for extension in ["_haps.npy","_pos.npy"]:
                    orig_file = os.path.join(from_dir, "{}{}".format(subsample_indices[i],extension))
                    new_file = os.path.join(to_dir, "{}{}".format(i,extension))
                    H = np.load(orig_file)
                    H = H[:minSites]
                    np.save(new_file,H)
        finally:
            task_q.task_done()
