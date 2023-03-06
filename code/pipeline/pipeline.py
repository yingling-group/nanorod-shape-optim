import sys
import traceback
import itertools
import random
from typing import Iterable

import pandas as pd

from . import utils

class Payload:
    """ Payload to pass and share between the Adapters of a GridLine.
    """
    def __init__(self, **kwargs):
        self.Tr = None
        self.Ts = None
        self.Tv = None
        self.xCols = []
        self.yCol = None
        self.xsclr = None 
        self.ysclr = None
        self.model = None
        self.cv = 5
        self.scoring = 'accuracy'
        self.score_report = None
        self.stats = {}
        self.score = 0

    def __repr__(self):
        t = ""
        if self.Tr is not None:
            t += "Tr shape: %s" %str(self.Tr.shape) + "\n"
            t += "Tr columns: %s" %(self.Tr.columns) + "\n"
        if self.Ts is not None:
            t += "Ts shape: %s" %str(self.Ts.shape) + "\n"
            t += "Ts columns: %s" %(self.Ts.columns) + "\n"
        if self.yCol is not None:
            t += "yCol: " + self.yCol + "\n"
        if self.xCols is not None:
            t += "xCols: " + str(list(self.xCols)) + "\n"
        if self.model is not None:
            t += "Model: " + str(self.model) + "\n"
            t += "Score: %0.2f" %self.score + "\n"
            t += "Stats: %s" %str(self.stats) + "\n"
        return t.rstrip()


class Adapter:
    """ The Adapter class to be overridden by each GridLine item.
        Use class variables of the Adapters for anything that should not be shared
        with other Adapters or to maintain a cache between pipelines.
    """
    def _newline(self, pipelineId = 0, stepId = 0):
        # Called on each new pipeline
        self.output = ""
        self.lineId = pipelineId
        self.stepId = stepId
        return self

    def __repr__(self):
        pkg = str(self.__class__).split("'")[1]
        if "." in pkg:
            pkg = pkg.split(".")[-1]
        return pkg + "()"

    def sayf(self, msg, *args, **kwargs):
        """ Store messages to print at last. """
        eol = "\n"
        if 'end' in kwargs:
            eol = kwargs['end']
        self.output += msg.format(*args) + eol

    def _report(self):
        if len(self.output) == 0:
            return None
        else:
            out = "\n" + self.output
            out = out.replace("\n", "\n\t ")
            return out.rstrip()

    def Process(self, X):
        raise NotImplementedError()
    
    def Execute(self, X, i = 0, s = 0, mute = False):
        print(" --", self, end = " ... ")
        sys.stdout.flush()

        self._newline(i, s)

        X = self.Process(X)
        rep = self._report()

        if rep:
            print(rep, end="\n\n")
        else:
            print("ok")

        return X

class GridLine:
    """ Define a list of Adapters, whose Process() methods will be
        called one after another with previous result being passed on. """
    def __init__(self, grid, muted = False):
        self.grid = []
        self.results = []
        self.adapters = {}
        self.muted = muted
        self.saveResults = None
        
        # convert to a list of lists
        for item in grid:
            if isinstance(item, Iterable):
                self.grid.append(item)
            else:
                self.grid.append([item])

        # make a list of all possible combinations
        self.grid = list(itertools.product(*self.grid))
        print("Initialized %d pipelines." %len(self.grid))

    def _pipeline(self, i, pipe, X):
        # Actual method to run a single pipeline
        
        for s, adapter in enumerate(pipe):
            if adapter is None:
                self.adapters['L%02d' %(i+1)].append("")
                continue

            assert isinstance(adapter, Adapter), \
                "Invalid adapter %s with type %s" %(str(adapter), type(adapter))
            
            self.adapters['L%02d' %(i+1)].append(utils.nice_name(adapter))

            try:
                X = adapter.Execute(X, i+1, s+1, self.muted)
            except Exception as err:
                traceback.print_exception(type(err), err, err.__traceback__)
                print('L%02d FAILED: %s' %(i+1, err))
                break
                
            assert isinstance(X, Payload), \
                "Invalid Process() return type. Payload expected."

        # final payload
        return X
    
    def ExecuteLine(self, lineNo, X):
        """ Run a single pipeline by it's number. """
        i = lineNo - 1
        pipe = self.grid[i]
        
        print("Pipeline %02d of %02d:" %(lineNo, len(self.grid)))
        print("================================================ >>")
        self.adapters['L%02d' %(lineNo)] = []
        if hasattr(X, 'copy'):
            xclone = X.copy()
        else:
            xclone = X

        newX = self._pipeline(i, pipe, xclone)
        print("Done %02d.\n" %(i+1))
        
        return newX
    
    def Shuffle(self):
        random.shuffle(self.grid)
    
    def Save(self, output_file = None):
        if output_file is not None:
            self.saveResults = output_file

        if self.saveResults is None:
            return
        
        if len(self.results) == 0:
            return

        df = self.Summarize()
        try:
            df.to_csv(self.saveResults)
        except Exception as err:
            # traceback.print_exception(type(err), err, err.__traceback__)
            print("Failed to save results to %s. %s" %(self.saveResults, err))
        

    def Execute(self, X):
        """ Run all the pipelines generated by `itertools.product()` """
        self.results = []
        for i in range(len(self.grid)):
            X = self.ExecuteLine(i+1, X)
            self.results.append(X)
            self.Save()

    def Scores(self):
        dc = {
            "score": [res.score for res in self.results],
            "model": [str(res.model) for res in self.results],
            "xcols": [" ".join(res.xCols) for res in self.results],
        }
        
        # Add columns for the stats
        for st in self.results[0].stats:
            dc[st] = []
            for res in self.results:
                if st in res.stats:
                    dc[st].append(res.stats[st])
                else:
                    dc[st].append("")

        df = pd.DataFrame(dc, index = [
            "L%02d" %(i+1) for i in range(len(self.results))]
        )
        df = df.sort_values('score', ascending=False)
        return df

    def Lines(self):
        ad = self.adapters
        
        # pad the shorter dicts, required if line fails
        l = max([len(ad[k]) for k in ad])
        for k in ad:
            d = l - len(ad[k])
            [ad[k].append("") for i in range(d)]
        return pd.DataFrame(ad,
                           index=["S%02d" %(i+1) for i in range(l)])

    def Summarize(self):
        return pd.concat([self.Scores().T, self.Lines()]).T

