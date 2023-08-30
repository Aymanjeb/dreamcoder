from collections import defaultdict
import datetime
import dill
import json
import math
import numpy as np
import os
import pickle
import random
import signal
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamcoder.dreaming import *
from dreamcoder.dreamcoder import explorationCompression, sleep_recognition
from dreamcoder.utilities import eprint, flatten, testTrainSplit, lse, runWithTimeout
from dreamcoder.grammar import Grammar, ContextualGrammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.recognition import RecognitionModel, DummyFeatureExtractor, variable
from dreamcoder.program import Program
from dreamcoder.domains.arc.utilsPostProcessing import *
from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.taskGeneration import *

DEFAULT_DOMAIN_NAME_PREFIX = "arc"
DEFAULT_LANGUAGE_DATASET_DIR = f"data/{DEFAULT_DOMAIN_NAME_PREFIX}/language"

class EvaluationTimeout(Exception):
    pass


class ArcTask(Task):
    def __init__(self, name, request, examples, evalExamples, features=None, cache=False):
        super().__init__(name, request, examples, features=features, cache=cache)
        self.evalExamples = evalExamples
           #added


    def checkEvalExamples(self, e, timeout=None):
        if timeout is not None:
            def timeoutCallBack(_1, _2): raise EvaluationTimeout()
        try:
            signal.signal(signal.SIGVTALRM, timeoutCallBack)
            signal.setitimer(signal.ITIMER_VIRTUAL, timeout)

            try:
                f = e.evaluate([])
            except IndexError:
                # free variable
                return False
            except Exception as e:
                eprint("Exception during evaluation:", e)
                return False

            for x, y in self.evalExamples:
                if self.cache and (x, e) in EVALUATIONTABLE:
                    p = EVALUATIONTABLE[(x, e)]
                else:
                    try:
                        p = self.predict(f, x)
                    except BaseException:
                        p = None
                    if self.cache:
                        EVALUATIONTABLE[(x, e)] = p
                if p != y:
                    if timeout is not None:
                        signal.signal(signal.SIGVTALRM, lambda *_: None)
                        signal.setitimer(signal.ITIMER_VIRTUAL, 0)
                    return False

            return True
        # except e:
            # eprint(e)
            # assert(False)
        except EvaluationTimeout:
            eprint("Timed out while evaluating", e)
            return False
        finally:
            if timeout is not None:
                signal.signal(signal.SIGVTALRM, lambda *_: None)
                signal.setitimer(signal.ITIMER_VIRTUAL, 0)

def retrieveARCJSONTasks(directory, augmented_dir=None, filenames=None):

    data = []

    for filename in os.listdir(directory):
        if ("json" in filename):
            task = retrieveARCJSONTask(filename, directory,augmented_dir=augmented_dir)
            if filenames is not None:
                if filename in filenames:
                    data.append(task)
            else:
                data.append(task)
    return data


def retrieveARCJSONTask(filename, directory,augmented_dir=None):
    with open(directory + "/" + filename, "r") as f:
        loaded = json.load(f)

    ioExamples = [
            ((Grid(gridArray=example["input"]),), Grid(gridArray=example["output"]))
            for example in loaded["train"]
        ]
    evalExamples = [
            ((Grid(gridArray=example["input"]),), Grid(gridArray=example["output"]))
            for example in loaded["test"]
        ]
    
    ######### I've added ################################################################################


    
    ######################################################################################################
    task = ArcTask(
        filename,
        arrow(tgridin, tgridout),
        ioExamples,
        evalExamples    # changed
    )
    task.specialTask = ('arc', 5)
    return task


def arc_options(parser):
    # parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--unigramEnumerationTimeout", type=int, default=3600)
    parser.add_argument("--firstTimeEnumerationTimeout", type=int, default=1)
    parser.add_argument("--featureExtractor", default="dummy", choices=[
        "arcCNN",
        "dummy"])

    parser.add_argument("--languageDatasetDir",             
        default=DEFAULT_LANGUAGE_DATASET_DIR,
        help="Top-level directory containing the language tasks.")


def check(filename, f, directory):
    train, test = retrieveARCJSONTask(filename, directory=directory)
    print(train)

    for input, output in train.examples:
        input = input[0]
        if f(input) == output:
            print("HIT")
        else:
            print("MISS")
            print("Got")
            f(input).pprint()
            print("Expected")
            output.pprint()

    return


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def gridToArray(grid):
    temp = np.full((grid.getNumRows(),grid.getNumCols()),None)
    for yPos,xPos in grid.points:
        temp[yPos, xPos] = str(grid.points[(yPos,xPos)])
    return temp

class ArcCNN(nn.Module):
    special = 'arc'
    
    def __init__(self, tasks=[], testingTasks=[], cuda=False, H=64, inputDimensions=25):
        super(ArcCNN, self).__init__()

        self.CUDA = cuda
        self.recomputeTasks = True

        self.outputDimensionality = H
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                #nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        self.gridDimension = 30

        # channels for hidden
        hid_dim = 64
        z_dim = 128

        self.encoder = nn.Sequential(
            conv_block(10,hid_dim),
            conv_block(hid_dim,hid_dim),
            conv_block(hid_dim,z_dim),
            nn.AdaptiveAvgPool2d((1, 1))
        )
            

    
    def forward(self, v):
        v_all = None
        assert v.shape == (v.shape[0], 22, self.gridDimension, self.gridDimension)
        v = variable(v, cuda=self.CUDA).float()
        for i in range(v.shape[0]):
            inputTensor= v[i,1:11,:int(v[i,0,0,0]),:int(v[i,0,0,1])]
            outputTensor = v[i,12:,:int(v[i,11,0,0]), :int(v[i,11,0,1])]
            v_input = inputTensor.unsqueeze(0)
            v_input = self.encoder(v_input)
            v_input = v_input.squeeze()
            v_output = outputTensor.unsqueeze(0)
            v_output = self.encoder(v_output)
            v_output = v_output.squeeze()
            ioTensor = torch.cat([v_input, v_output], 0).unsqueeze(0)
            if v_all is None:
                v_all = ioTensor
            else:
                v_all = torch.cat([v_all, ioTensor], dim=0)
        v_all = v_all.squeeze()
        device = torch.device('cuda')
        v_all = v_all.to(device)
        linear_layer = nn.Linear(256,64).to(device)
        v_all = linear_layer(v_all)
        v_all = nn.ReLU()(v_all)
        v_all = v_all.mean(dim=0)
        return v_all


    def featuresOfTask(self, t):  # Take a task and returns [features]
        v = None
        for example in t.examples:
            inputGrid, outputGrid = example
            inputGrid = inputGrid[0]
            inputTensor = inputGrid.to_tensor(inputGrid.getNumRows(), inputGrid.getNumCols())
            outputTensor = outputGrid.to_tensor(outputGrid.getNumRows(), outputGrid.getNumCols())
            inputTensor = inputGrid.to_tensor(grid_height=30, grid_width=30)
            outputTensor = outputGrid.to_tensor(grid_height=30, grid_width=30)
            #added code
            inputTensor[0,0,0], inputTensor[0,0,1] = inputGrid.getNumRows(), inputGrid.getNumCols()
            outputTensor[0,0,0], outputTensor[0,0,1] = outputGrid.getNumRows(), outputGrid  .getNumCols()
            ####
            ioTensor = torch.cat([inputTensor, outputTensor], 0).unsqueeze(0)

            if v is None:
                v = ioTensor
            else:
                v = torch.cat([v, ioTensor], dim=0)
        return self(v)
    
    def taskOfProgram(self, p, tp):
        """
        For simplicitly we only use one example per task randomly sampled from
        all possible input grids we've seen.
        """
        def randomInput(t): return random.choice(self.argumentsWithType[t])

        startTime = time.time()
        examples = []
        while True:
            # TIMEOUT! this must not be a very good program
            if time.time() - startTime > self.helmholtzTimeout: return None

            # Grab some random inputs
            xs = [randomInput(t) for t in tp.functionArguments()]
            try:
                y = runWithTimeout(lambda: p.runWithArguments(xs), self.helmholtzEvaluationTimeout)
                examples.append((tuple(xs),y))
                if len(examples) >= 1:
                    return Task("Helmholtz", tp, examples)
            except: continue
        return None


    def featuresOfTasks(self, ts, t2=None):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        return [self.featuresOfTask(t) for t in ts]





def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on manipulating sequences of numbers.

    """


    import os

    homeDirectory = "/".join(os.path.abspath(__file__).split("/")[:-4])
    dataDirectory = homeDirectory + "/arc_data/data/"


    #trainTasks = retrieveARCJSONTasks(dataDirectory + 'all_data', None)
    trainTasks = retrieveARCJSONTasks('/home/jebari/ARC/new_version/arc_data/data/training')
    #holdoutTasks = retrieveARCJSONTasks(dataDirectory + 'evaluation')

    baseGrammar = Grammar.uniform(basePrimitives() + leafPrimitives())
    # print("base Grammar {}".format(baseGrammar))

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/arc/%s" % timestamp
    os.system("mkdir -p %s" % outputDirectory)

    args.update(
        {"outputPrefix": "%s/arc" % outputDirectory, "evaluationTimeout": 1,}
    )

    featureExtractor = {
        "dummy": DummyFeatureExtractor,
        "arcCNN": ArcCNN
    }[args.pop("featureExtractor")]
    
    explorationCompression(baseGrammar, trainTasks, featureExtractor=featureExtractor, testingTasks=[], **args)
