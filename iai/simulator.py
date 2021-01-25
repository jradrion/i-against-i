'''
Author: Jared Galloway, Jeff Adrion
'''

from iai.imports import *
from iai.helpers import *

class Simulator(object):
    '''

    The simulator class is a framework for running N simulations
    using Either msprime (coalescent) or SLiM (forward-moving)
    in parallel using python's multithreading package.

    With Specified parameters, the class Simulator() populates
    a directory with training, validation, and testing datasets.
    It stores the the treeSequences resulting from each simulation
    in a subdirectory respectfully labeled 'i.trees' where i is the
    i^th simulation.

    Included with each dataset this class produces an info.p
    in the subdirectory. This uses pickle to store a dictionary
    containing all the information for each simulation including the random
    target parameter which will be extracted for training.

    '''

    def __init__(self,
        N = 2,
	Ne = 1e2,
        Ne_growth_lo = 1e2,
        Ne_growth_hi = 1e2,
        Ne_noGrowth = 1e2,
        priorLowsRho = 0.0,
        priorLowsMu = 0.0,
        priorHighsRho = 1e-7,
        priorHighsMu = 1e-8,
        priorLowsGr = 0.0,
        priorHighsGr = 1e-8,
        priorLowsM = 0.0,
        priorHighsM = 1e-7,
        fractionGrowth = 0.5,
        fractionAdmix = 0.5,
        ChromosomeLength = 1e5,
        expansion = None,
        admixture = None,
        MspDemographics = None,
        winMasks = None,
        mdMask = None,
        maskThresh = 1.0,
        phased = None,
        phaseError = None,
        seed = None,
        bn = None,
        testGrid = 0,
        gridParams = None,
        gridPars = None
        ):

        self.N = N
        self.Ne = Ne
        self.Ne_growth_lo = Ne_growth_lo
        self.Ne_growth_hi = Ne_growth_hi
        self.Ne_noGrowth = Ne_noGrowth
        self.priorLowsRho = priorLowsRho
        self.priorHighsRho = priorHighsRho
        self.priorLowsMu = priorLowsMu
        self.priorHighsMu = priorHighsMu
        self.priorLowsGr = priorLowsGr
        self.priorHighsGr = priorHighsGr
        self.priorLowsM = priorLowsM
        self.priorHighsM = priorHighsM
        self.fractionGrowth = fractionGrowth
        self.fractionAdmix = fractionAdmix
        self.ChromosomeLength = ChromosomeLength
        self.MspDemographics = MspDemographics
        self.rho = None
        self.mu = None
        self.m = None
        self.segSites = None
        self.winMasks = winMasks
        self.mdMask = mdMask
        self.maskThresh = maskThresh
        self.phased = None
        self.expansion = expansion
        self.admixture = admixture
        self.phaseError = phaseError
        self.seed = seed
        self.testGrid = testGrid
        self.gridPars = gridPars
        if gridParams:
            self.gridParams = gridParams.split(",")
        else:
            self.gridParams = []


    def runOneMsprimeSim(self,simNum,direc):
        '''
        run one msprime simulation and put the corresponding treeSequence in treesOutputFilePath

        (str,float,float)->None
        '''

        MR = self.mu[simNum]
        RR = self.rho[simNum]
        GR = self.gr[simNum]
        MIG = self.m[simNum]
        NE = self.ne[simNum]
        SEED = self.seed[simNum]

        #GR = GR * -1.0 ######remove after testing pop crash

        if self.expansion and not self.admixture:
            if GR > 0.0:
                PC = [msp.PopulationConfiguration(
                    sample_size=self.N,
                    initial_size=NE,
                    growth_rate=GR)]

                ts = msp.simulate(
                    random_seed = SEED,
                    length=self.ChromosomeLength,
                    mutation_rate=MR,
                    recombination_rate=RR,
                    population_configurations = PC,
                )
            else:
                ts = msp.simulate(
                    random_seed = SEED,
                    sample_size = self.N,
                    Ne = self.Ne_noGrowth,
                    length=self.ChromosomeLength,
                    mutation_rate=MR,
                    recombination_rate=RR
                )
        elif self.admixture and not self.expansion:
            if MIG > 0.0:
                PC = [
                        msp.PopulationConfiguration(
                            sample_size=self.N,
                            initial_size=NE),

                        msp.PopulationConfiguration(
                            sample_size=self.N,
                            initial_size=NE)
                        ]

                MM = [[      0, MIG],
                        [MIG,       0]]

                DE = []
                DD = msp.DemographyDebugger(
                        population_configurations=PC,
                        migration_matrix=MM,
                        demographic_events=DE)
                #DD.print_history()
                #start = time.time()
                ts = msp.simulate(
                    random_seed = SEED,
                    length=self.ChromosomeLength,
                    mutation_rate=MR,
                    recombination_rate=RR,
                    population_configurations = PC,
                    migration_matrix = MM,
                    demographic_events = DE

                )
                #end = time.time()
                #print("sim_completed in :",end-start)
            else:
                ts = msp.simulate(
                    random_seed = SEED,
                    sample_size = self.N,
                    Ne = self.Ne_noGrowth,
                    length=self.ChromosomeLength,
                    mutation_rate=MR,
                    recombination_rate=RR
                )
        elif self.admixture and self.expansion:
            if MIG > 0.0 and GR > 0.0:
                PC = [
                        msp.PopulationConfiguration(
                            sample_size=self.N,
                            initial_size=NE,
                            growth_rate=GR),

                        msp.PopulationConfiguration(
                            sample_size=self.N,
                            initial_size=NE,
                            growth_rate=GR)
                        ]

                MM = [[      0, MIG],
                        [MIG,       0]]

                DE = []
                DD = msp.DemographyDebugger(
                        population_configurations=PC,
                        migration_matrix=MM,
                        demographic_events=DE)
                ts = msp.simulate(
                    random_seed = SEED,
                    length=self.ChromosomeLength,
                    mutation_rate=MR,
                    recombination_rate=RR,
                    population_configurations = PC,
                    migration_matrix = MM,
                    demographic_events = DE

                )
            elif MIG > 0.0 and GR == 0.0:
                PC = [
                        msp.PopulationConfiguration(
                            sample_size=self.N,
                            initial_size=NE),

                        msp.PopulationConfiguration(
                            sample_size=self.N,
                            initial_size=NE)
                        ]

                MM = [[      0, MIG],
                        [MIG,       0]]

                DE = []
                DD = msp.DemographyDebugger(
                        population_configurations=PC,
                        migration_matrix=MM,
                        demographic_events=DE)
                ts = msp.simulate(
                    random_seed = SEED,
                    length=self.ChromosomeLength,
                    mutation_rate=MR,
                    recombination_rate=RR,
                    population_configurations = PC,
                    migration_matrix = MM,
                    demographic_events = DE

                )
            elif GR > 0.0 and MIG == 0.0:
                PC = [msp.PopulationConfiguration(
                    sample_size=self.N,
                    initial_size=NE,
                    growth_rate=GR)]

                ts = msp.simulate(
                    random_seed = SEED,
                    length=self.ChromosomeLength,
                    mutation_rate=MR,
                    recombination_rate=RR,
                    population_configurations = PC,
                )
            else:
                ts = msp.simulate(
                    random_seed = SEED,
                    sample_size = self.N,
                    Ne = self.Ne_noGrowth,
                    length=self.ChromosomeLength,
                    mutation_rate=MR,
                    recombination_rate=RR
                )
        else:
            ts = msp.simulate(
                random_seed = SEED,
                sample_size = self.N,
                Ne = self.Ne_noGrowth,
                length=self.ChromosomeLength,
                mutation_rate=MR,
                recombination_rate=RR
            )


        # Convert tree sequence to genotype matrix, and position matrix
        if self.admixture: # for migration matrix simulations with two popuulations, extract variant matrix from population 0 only
            #start = time.time()
            ts_pop0 = ts.simplify(ts.samples(population=0))
            H_pop0 = ts_pop0.genotype_matrix()
            P_pop0 = np.array([s.position for s in ts_pop0.sites()],dtype='float32')

            # mask all invariant sites
            zero_mask = ~np.all(H_pop0 == 0,axis=1)
            H_pop0 = H_pop0[zero_mask]
            P_pop0 = P_pop0[zero_mask]
            one_mask = ~np.all(H_pop0 == 1,axis=1)
            H = H_pop0[one_mask]
            P = P_pop0[one_mask]
            #end = time.time()
            #print("simplify time:",end-start)
        else:
            H = ts.genotype_matrix()
            P = np.array([s.position for s in ts.sites()],dtype='float32')


        # "Unphase" genotypes
        if not self.phased:
            np.random.shuffle(np.transpose(H))

        # Simulate phasing error
        if self.phaseError:
            H = self.phaseErrorer(H,self.phaseError)

        # If there is a missing data mask, sample from the mask and apply to haps
        if not self.mdMask is None:
            mdMask = self.mdMask[np.random.choice(self.mdMask.shape[0], H.shape[0], replace=True)]
            H = np.ma.masked_array(H, mask=mdMask)
            H = np.ma.filled(H,2)

        # Sample from the genome-wide distribution of masks and mask both positions and genotypes
        if self.winMasks:
            while True:
                rand_mask = self.winMasks[random.randint(0,len(self.winMasks)-1)]
                if rand_mask[0] < self.maskThresh:
                    break
            if rand_mask[0] > 0.0:
                H,P = self.maskGenotypes(H, P, rand_mask)

        # Dump
        Hname = str(simNum) + "_haps.npy"
        Hpath = os.path.join(direc,Hname)
        Pname = str(simNum) + "_pos.npy"
        Ppath = os.path.join(direc,Pname)
        np.save(Hpath,H)
        np.save(Ppath,P)

        # Return number of sites
        return H.shape[0]


    def maskGenotypes(self, H, P, rand_mask):
        """
        Return the genotype and position matrices where masked sites have been removed
        """
        mask_wins = np.array(rand_mask[1])
        mask_wins = np.reshape(mask_wins, 2 * mask_wins.shape[0])
        mask = np.digitize(P, mask_wins) % 2 == 0
        return H[mask], P[mask]


    def phaseErrorer(self, H, rate):
        """
        Returns the genotype matrix where some fraction of sites have shuffled samples
        """
        H_shuf = copy.deepcopy(H)
        np.random.shuffle(np.transpose(H_shuf))
        H_mask = np.random.choice([True,False], H.shape[0], p = [1-rate,rate])
        H_mask = np.repeat(H_mask, H.shape[1])
        H_mask = H_mask.reshape(H.shape)
        return np.where(H_mask,H,H_shuf)


    def simulateAndProduceTrees(self,direc,numReps,simulator,nProc=1,
            X=None, Y=None, Z=None, gridPars=None):
        '''
        determine which simulator to use then populate

        (str,str) -> None
        '''

        if "gr" in self.gridParams and "mu" in self.gridParams:
            self.gr = np.linspace(0.0,self.priorHighsGr,self.testGrid)
            self.mu = np.linspace(self.priorLowsMu,self.priorHighsMu,self.testGrid)
            self.gr, self.mu = np.meshgrid(self.gr,self.mu)
            self.gr = np.repeat(self.gr,numReps)
            self.mu = np.repeat(self.mu,numReps)
            self.numReps = self.gr.shape[0]
            numReps = self.numReps
            self.ne=np.zeros(numReps)
            for i in range(numReps):
                randomTargetParameter = np.random.uniform(self.Ne_growth_lo,self.Ne_growth_hi)
                self.ne[i] = randomTargetParameter
            self.rho=np.empty(numReps)
            for i in range(numReps):
                randomTargetParameter = np.random.uniform(self.priorLowsRho,self.priorHighsRho)
                self.rho[i] = randomTargetParameter
            self.seed=np.repeat(self.seed,numReps)

        elif "gr" in self.gridParams and "ne" in self.gridParams:
            self.gr = np.linspace(0.0,self.priorHighsGr,self.testGrid)
            self.ne = np.linspace(self.Ne_growth_lo,self.Ne_growth_hi,self.testGrid)
            self.gr, self.ne = np.meshgrid(self.gr,self.ne)
            self.gr = np.repeat(self.gr,numReps)
            self.ne = np.repeat(self.ne,numReps)
            self.numReps = self.gr.shape[0]
            numReps = self.numReps
            self.mu=np.empty(numReps)
            for i in range(numReps):
                randomTargetParameter = np.random.uniform(self.priorLowsMu,self.priorHighsMu)
                self.mu[i] = randomTargetParameter
            self.rho=np.empty(numReps)
            for i in range(numReps):
                randomTargetParameter = np.random.uniform(self.priorLowsRho,self.priorHighsRho)
                self.rho[i] = randomTargetParameter
            self.seed=np.repeat(self.seed,numReps)

        elif "gr" in self.gridParams and "rho" in self.gridParams:
            self.gr = np.linspace(0.0,self.priorHighsGr,self.testGrid)
            self.rho = np.linspace(self.priorLowsRho,self.priorHighsRho,self.testGrid)
            self.gr, self.rho = np.meshgrid(self.gr,self.rho)
            self.gr = np.repeat(self.gr,numReps)
            self.rho = np.repeat(self.rho,numReps)
            self.numReps = self.gr.shape[0]
            numReps = self.numReps
            self.mu=np.empty(numReps)
            for i in range(numReps):
                randomTargetParameter = np.random.uniform(self.priorLowsMu,self.priorHighsMu)
                self.mu[i] = randomTargetParameter
            self.ne=np.zeros(numReps)
            for i in range(numReps):
                randomTargetParameter = np.random.uniform(self.Ne_growth_lo,self.Ne_growth_hi)
                self.ne[i] = randomTargetParameter
            self.seed=np.repeat(self.seed,numReps)

        elif "mu" in self.gridParams and "ne" in self.gridParams:
            self.mu = np.linspace(self.priorLowsMu,self.priorHighsMu,self.testGrid)
            self.ne = np.linspace(self.Ne_growth_lo,self.Ne_growth_hi,self.testGrid)
            self.mu, self.ne = np.meshgrid(self.mu,self.ne)
            self.mu = np.repeat(self.mu,numReps)
            self.ne = np.repeat(self.ne,numReps)
            self.numReps = self.mu.shape[0]
            numReps = self.numReps
            self.gr=np.zeros(numReps)
            for i in range(int(numReps*self.fractionGrowth)):
                randomTargetParameter = np.random.uniform(self.priorLowsGr,self.priorHighsGr)
                self.gr[i] = randomTargetParameter
            self.rho=np.empty(numReps)
            for i in range(numReps):
                randomTargetParameter = np.random.uniform(self.priorLowsRho,self.priorHighsRho)
                self.rho[i] = randomTargetParameter
            self.seed=np.repeat(self.seed,numReps)

        elif "mu" in self.gridParams and "rho" in self.gridParams:
            self.mu = np.linspace(self.priorLowsMu,self.priorHighsMu,self.testGrid)
            self.rho = np.linspace(self.priorLowsRho,self.priorHighsRho,self.testGrid)
            self.mu, self.rho = np.meshgrid(self.mu,self.rho)
            self.mu = np.repeat(self.mu,numReps)
            self.rho = np.repeat(self.rho,numReps)
            self.numReps = self.mu.shape[0]
            numReps = self.numReps
            self.gr=np.zeros(numReps)
            for i in range(int(numReps*self.fractionGrowth)):
                randomTargetParameter = np.random.uniform(self.priorLowsGr,self.priorHighsGr)
                self.gr[i] = randomTargetParameter
            self.ne=np.zeros(numReps)
            for i in range(numReps):
                randomTargetParameter = np.random.uniform(self.Ne_growth_lo,self.Ne_growth_hi)
                self.ne[i] = randomTargetParameter
            self.seed=np.repeat(self.seed,numReps)

        elif "ne" in self.gridParams and "rho" in self.gridParams:
            self.ne = np.linspace(self.Ne_growth_lo,self.Ne_growth_hi,self.testGrid)
            self.rho = np.linspace(self.priorLowsRho,self.priorHighsRho,self.testGrid)
            self.ne, self.rho = np.meshgrid(self.ne,self.rho)
            self.ne = np.repeat(self.ne,numReps)
            self.rho = np.repeat(self.rho,numReps)
            self.numReps = self.ne.shape[0]
            numReps = self.numReps
            self.gr=np.zeros(numReps)
            for i in range(int(numReps*self.fractionGrowth)):
                randomTargetParameter = np.random.uniform(self.priorLowsGr,self.priorHighsGr)
                self.gr[i] = randomTargetParameter
            self.mu=np.empty(numReps)
            for i in range(numReps):
                randomTargetParameter = np.random.uniform(self.priorLowsMu,self.priorHighsMu)
                self.mu[i] = randomTargetParameter
            self.seed=np.repeat(self.seed,numReps)

        else:
            self.gr=np.zeros(numReps)
            for i in range(int(numReps*self.fractionGrowth)):
                randomTargetParameter = np.random.uniform(self.priorLowsGr,self.priorHighsGr)
                self.gr[i] = randomTargetParameter
            self.m=np.zeros(numReps)
            for i in range(int(numReps*self.fractionAdmix)):
                randomTargetParameter = np.random.uniform(self.priorLowsM,self.priorHighsM)
                self.m[i] = randomTargetParameter
            ### restrict area of parameter space for positive control
            #for i in range(int(numReps*self.fractionGrowth)):
            #    randomTargetParameter = 5e-6
            #    while 2e-6 <= randomTargetParameter <= 6e-6:
            #        randomTargetParameter = np.random.uniform(self.priorLowsGr,self.priorHighsGr)
            #    self.gr[i] = randomTargetParameter
            ### add constant to an area of parameter space

            #self.gr[0] = 1e-9
            #print("migration rate:",self.gr)
            #return
            self.mu=np.empty(numReps)
            for i in range(numReps):
                randomTargetParameter = np.random.uniform(self.priorLowsMu,self.priorHighsMu)
                self.mu[i] = randomTargetParameter
            ### restrict area of parameter space for positive control
            #for i in range(numReps):
            #    randomTargetParameter = 5e-7
            #    while 2e-7 <= randomTargetParameter <= 6e-7:
            #        randomTargetParameter = np.random.uniform(self.priorLowsMu,self.priorHighsMu)
            #    self.mu[i] = randomTargetParameter

            self.ne=np.zeros(numReps)
            for i in range(numReps):
                randomTargetParameter = np.random.uniform(self.Ne_growth_lo,self.Ne_growth_hi)
                self.ne[i] = randomTargetParameter
            self.rho=np.empty(numReps)
            for i in range(numReps):
                randomTargetParameter = np.random.uniform(self.priorLowsRho,self.priorHighsRho)
                self.rho[i] = randomTargetParameter
            self.seed=np.repeat(self.seed,numReps)


            # Plot histograms of the parameters
            nbins = 100
            if direc.split("/")[-1] == "train":
                plt.figure(0)
                outFile = os.path.join("/".join(d for d in direc.split("/")[:-1]),"train_mu.png")
                fig = plt.hist(self.mu,
                        range=(self.priorLowsMu,self.priorHighsMu),
                        bins=nbins)
                plt.savefig(outFile)
                plt.figure(1)
                outFile = os.path.join("/".join(d for d in direc.split("/")[:-1]),"train_ne.png")
                fig = plt.hist(self.ne,
                        range=(self.Ne_growth_lo,self.Ne_growth_hi),
                        bins=nbins)
                plt.savefig(outFile)
                plt.figure(2)
                outFile = os.path.join("/".join(d for d in direc.split("/")[:-1]),"train_rho.png")
                fig = plt.hist(self.rho,
                        range=(self.priorLowsRho,self.priorHighsRho),
                        bins=nbins)
                plt.savefig(outFile)
                if self.expansion:
                    plt.figure(3)
                    outFile = os.path.join("/".join(d for d in direc.split("/")[:-1]),"train_gr.png")
                    fig = plt.hist(self.gr,
                            range=(0.0,self.priorHighsGr),
                            bins=nbins)
                    plt.savefig(outFile)
                if self.admixture:
                    plt.figure(4)
                    outFile = os.path.join("/".join(d for d in direc.split("/")[:-1]),"train_mig.png")
                    fig = plt.hist(self.m,
                            range=(0.0,self.priorHighsM),
                            bins=nbins)
                    plt.savefig(outFile)



        # for adaptive training, set parameters to those of the worst predicted examples in the test set
        if type(Z) is np.ndarray:
            ct = 0
            x, y = np.zeros(numReps), np.zeros(numReps)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    for k in range(int(Z[i][j])):
                        randomTargetParameter = np.random.uniform(X[i][max(0,j-1)],
                                X[i][min(X.shape[1]-1,j+1)])
                        x[ct] = randomTargetParameter
                        randomTargetParameter = np.random.uniform(Y[max(0,i-1)][j],
                                Y[min(X.shape[0]-1,i+1)][j])
                        y[ct] = randomTargetParameter
                        ct+=1

            for i in range(len(gridPars)):
                if gridPars[i] == "gr":
                    if i == 0:
                        self.gr = x
                    else:
                        self.gr = y
                    print("self.gr:",self.gr)
                    ## randomly mask set 'fractionGrowth' of the gr values to zero
                    #mask = np.zeros(self.gr.shape[0], dtype=int)
                    #mask[:int(mask.shape[0]*self.fractionGrowth)] = 1
                    #np.random.shuffle(mask)
                    #mask = mask.astype(bool)
                    #self.gr = np.where(mask == True, self.gr, 0.0)

                    ## don't set 'fractionGrowth' to zero but do set values under 1e-8, otherwise they take too long to coalesce.
                    self.gr[self.gr < 1e-8] = 0.0
                    print("new_gr:",self.gr)
                    #return
                if gridPars[i] == "mu":
                    if i == 0:
                        self.mu = x
                    else:
                        self.mu = y
                if gridPars[i] == "ne":
                    if i == 0:
                        self.ne = x
                    else:
                        self.ne = y
                if gridPars[i] == "rho":
                    if i == 0:
                        self.rho = x
                    else:
                        self.rho = y
            self.seed = np.multiply(np.arange(1,numReps+1),self.seed)


            # Plot histograms of the parameters
            nbins = 100
            if direc.split("/")[-1] == "train_adapt":
                plt.figure(10)
                outFile = os.path.join("/".join(d for d in direc.split("/")[:-1]),"train_adapt_gr.png")
                fig = plt.hist(self.gr,
                        range=(0.0,self.priorHighsGr),
                        bins=nbins)
                plt.savefig(outFile)
                plt.figure(11)
                outFile = os.path.join("/".join(d for d in direc.split("/")[:-1]),"train_adapt_mu.png")
                fig = plt.hist(self.mu,
                        range=(self.priorLowsMu,self.priorHighsMu),
                        bins=nbins)
                plt.savefig(outFile)
                plt.figure(12)
                outFile = os.path.join("/".join(d for d in direc.split("/")[:-1]),"train_adapt_ne.png")
                fig = plt.hist(self.ne,
                        range=(self.Ne_growth_lo,self.Ne_growth_hi),
                        bins=nbins)
                plt.savefig(outFile)
                plt.figure(13)
                outFile = os.path.join("/".join(d for d in direc.split("/")[:-1]),"train_adapt_rho.png")
                fig = plt.hist(self.rho,
                        range=(self.priorLowsRho,self.priorHighsRho),
                        bins=nbins)
                plt.savefig(outFile)


        try:
            assert((simulator=='msprime') | (simulator=='SLiM'))
        except:
            print("Sorry, only 'msprime' & 'SLiM' are supported simulators")
            exit()

        #Pretty straitforward, create the directory passed if it doesn't exits
        if not os.path.exists(direc):
            print("directory '",direc,"' does not exist, creating it")
            os.makedirs(direc)

        # partition data for multiprocessing
        mpID = range(numReps)
        task_q = mp.JoinableQueue()
        result_q = mp.Queue()
        params=[simulator, direc]

        # do the work
        print("Simulate...")
        pids = create_procs(nProc, task_q, result_q, params, self.worker_simulate)
        assign_task(mpID, task_q, nProc)
        try:
            task_q.join()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(0)

        self.segSites=np.empty(numReps,dtype="int64")
        for i in range(result_q.qsize()):
            item = result_q.get()
            self.segSites[item[0]]=item[1]

        #gr,no_gr=[],[]
        #for i in range(len(self.gr)):
        #    if self.gr[i] == 0.0:
        #        no_gr.append(self.segSites[i])
        #    else:
        #        gr.append(self.segSites[i])

        #if gr:
        #    if self.admixture:
        #        print("mean segSites with expansion/admixture:", sum(gr)/float(len(gr)))
        #    else:
        #        print("mean segSites with expansion:", sum(gr)/float(len(gr)))
        #if no_gr:
        #    if self.admixture:
        #        print("mean segSites no admixture:", sum(no_gr)/float(len(no_gr)))
        #    else:
        #        print("mean segSites no growth:", sum(no_gr)/float(len(no_gr)))

        self.__dict__["numReps"] = numReps
        if self.expansion:
            self.__dict__["expansion"] = True
        else:
            self.__dict__["expansion"] = False
        if self.admixture:
            self.__dict__["admixture"] = True
        else:
            self.__dict__["admixture"] = False
        infofile = open(os.path.join(direc,"info.p"),"wb")
        pickle.dump(self.__dict__,infofile)
        infofile.close()

        for p in pids:
            p.terminate()
        return None


    def worker_simulate(self, task_q, result_q, params):
        while True:
            try:
                mpID, nth_job = task_q.get()
                #unpack parameters
                simulator, direc = params
                for i in mpID:
                        result_q.put([i,self.runOneMsprimeSim(i,direc)])
            finally:
                task_q.task_done()
