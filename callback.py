import json
import keras.backend as kb
import numpy as np
import os
import keras
import shutil
import warnings
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score,accuracy_score
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
# pathneg = pd.read_csv('./csv/negativepathall801.csv')
# pathpos = pd.read_csv('./csv/positivepathall801.csv')
# pathtrain = pd.concat([pathneg[:600],pathpos[:480]])
# pathval = pd.concat([pathneg[600:900],pathpos[480:]])

class MultipleClassAUROC(Callback):
    """
    Monitor mean AUROC and update model
    """
    def __init__(self,pathtrain,df,saveriskpath,sequence,sequence1, class_names, weights_path, stats=None, workers=1, log_path='./'):
        super(Callback, self).__init__()
        self.sequence = sequence
        self.sequence1 = sequence1
        self.workers = workers
        self.pathtrain= pathtrain
        self.df = df
        self.saveriskpath = saveriskpath
        self.class_names = len(class_names)
        self.weights_path = weights_path
        self.log_path = os.path.join(log_path, 'accAUCinTest.log')
        self.best_weights_path = os.path.join(
            os.path.split(weights_path)[0],
            f"best_{os.path.split(weights_path)[1]}",
        )
        self.best_auroc_log_path = os.path.join(
            os.path.split(weights_path)[0],
            "best_auroc.log",
        )
        self.stats_output_path = os.path.join(
            os.path.split(weights_path)[0],
            ".training_stats.json"
        )
        # for resuming previous training
        if stats:
            self.stats = stats
        else:
            self.stats = {"best_mean_auroc": 0}
        self.stats = {"best_mean_auroc": 0}
        # aurocs log
        self.aurocs = {}
        if self.class_names>2:
            for c in self.class_names:
                self.aurocs[c] = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate the average AUROC and save the best model weights according
        to this metric.

        """
        print("\n*********************************")
        self.stats["lr"] = float(kb.eval(self.model.optimizer.lr))
        print(f"current learning rate: {self.stats['lr']}")

        """
        y_hat shape: (#samples, len(class_names))
        y: [(#samples, 1), (#samples, 1) ... (#samples, 1)]
        """
        y_hat = self.model.predict_generator(self.sequence, workers=self.workers)
        y_hat1 = self.model.predict_generator(self.sequence1, workers=self.workers)
        # y = self.sequence.get_y_true()
        y = self.sequence.classes
        y1 = self.df['class']
        y1 = self.sequence1.classes
        print(f"*** epoch#{epoch + 1} dev auroc ***")
        current_auroc = []
        # for i in range(len(self.class_names)):
        #     try:
        score = roc_auc_score(y, y_hat)
            # except ValueError:
            #     score = 0
        # self.aurocs[self.class_names[0]].append(score)
        current_auroc.append(score)
        accval = accuracy_score(y, y_hat>0.5)
        auctest = roc_auc_score(y1, y_hat1)
        acctest = accuracy_score(y1, y_hat1>0.5)
            # except ValueError:
            #     score = 0
        # self.aurocs[self.class_names[0]].append(score)
        # print(f"{1}. {self.class_names}: {score}")
        print(f"acc validation is {accval:.3f}")
        # customize your multiple class metrics here
        mean_auroc = np.mean(current_auroc)
        print(f"auc validation is: {mean_auroc:.3f}")
        if mean_auroc > self.stats["best_mean_auroc"]:
            print(f"update best auroc from {self.stats['best_mean_auroc']:.3f} to {mean_auroc:.3f}")

            # 1. copy best model
            # shutil.copy(self.weights_path, self.best_weights_path)

            # 2. update log file
            print(f"update log file: {self.best_auroc_log_path}")
            with open(self.best_auroc_log_path, "a") as f:
                f.write(f"(epoch#{epoch + 1}) auroc: {mean_auroc:.3f}, lr: {self.stats['lr']}\n")

            # 3. write stats output, this is used for resuming the training
            with open(self.stats_output_path, 'w') as f:
                json.dump(self.stats, f)

            print(f"update model file: {self.weights_path} -> {self.best_weights_path}")
            self.stats["best_mean_auroc"] = mean_auroc
            print("********************")
        auctest = roc_auc_score(y1, y_hat1)
        acctest = accuracy_score(y1, y_hat1>0.5)
            # except ValueError:
            #     score = 0
        # self.aurocs[self.class_names[0]].append(score)
        # print(f"{1}. {self.class_names}: {score}")
        print("***Calculate for test***")
        print(f"\n auctest is {auctest:.3f}")
        print(f"acctest is {acctest:.3f}")
        print("**********************")
        with open(self.log_path, "a") as f:
          f.write(f"(epoch#{epoch + 1}) aurocval: {mean_auroc:.3f}, auctest: {auctest:.3f},accval:{accval:.3f} acctest: {acctest:.3f}\n")
        '''
        test_datagen=ImageDataGenerator(rescale=1./255.)
        train_generator=test_datagen.flow_from_dataframe(
                            dataframe=self.pathtrain,
                            directory=None,
                            x_col="originpath",
                            y_col="label",
                            # subset="training",
                            batch_size=32,
                            seed=1,
                            shuffle=False,
                            class_mode="binary",#if more than one class, here should be categorical
                            target_size=(224,224))

        test_generator=test_datagen.flow_from_dataframe(
                        dataframe=self.df,
                        directory=None,
                        x_col="originpath",
                        y_col='label',
                        batch_size=32,
                        seed=42,
                        shuffle=False,
                        class_mode=None,
                        target_size=(224,224))
        for i,se in enumerate([train_generator, test_generator]):#,self.test_sequence
            y_hat = self.model.predict_generator(se, workers=1)
            
            y = se.classes if i == 0 else self.df['class']
            filepath = se.filepaths
            pid=[]
            for each in filepath:
                patientdirnum = each.split('/')[3]
                if each.split('/')[2]=='negative':
                    id = patientdirnum.split('_')[0]+'neg'
                else:
                    id = patientdirnum.split('_')[0]+'pos'
                pid.append(id)
            pid = np.array(pid)
            res_all = pd.DataFrame(dict(PatientID=pid, label=y, Risk=y_hat.squeeze()))
            res = pd.DataFrame([[pid_, group.label.mean(), group.Risk.mean()] for pid_, group in res_all.groupby(by='PatientID')], columns=list(res_all))
            dataset = 'training'if i==0 else 'test'
            res_all.to_csv(os.path.join( self.saveriskpath,f'{epoch}_all_{dataset}926P1.csv'), index=False)
            res.to_csv(os.path.join( self.saveriskpath,f'{epoch}_{dataset}926P1.csv'), index=False)
        '''
        return