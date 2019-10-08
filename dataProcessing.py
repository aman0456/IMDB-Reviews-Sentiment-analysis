import pandas as pd
import random
import os

BASEPATH = "/Users/Aman/Work/ImdbReviewSentimentAnalysis"
DATA_DIR = "aclImdb"
TRAIN_DIR = DATA_DIR + "/train"
TEST_DIR = DATA_DIR + "/test"

filedata = []

def loadFiles(cdir, label):
	global filedata
	filenames = [os.path.join(cdir, fname) for fname in os.listdir(cdir)]
	def fread(filename):
		with open(filename, "r") as file:
			return file.read().replace(r'\n', ' '), label
	filedata += list(map(fread, filenames))

def getTSV(cdir, cstr):
	global filedata
	filedata = []
	loadFiles(os.path.join(BASEPATH, cdir, "pos"), 1)
	loadFiles(os.path.join(BASEPATH, cdir, "neg"), 0)

	random.shuffle(filedata)
	fileTexts = [item for item,_ in filedata]
	fileLabels = [item for _,item in filedata]

	assert len(fileTexts) == 25000 and len(fileTexts) == len(fileLabels)

	tbert = pd.DataFrame({
			'id' : range(len(fileTexts)),
			'label' : fileLabels,
			'alpha' : ['a']*len(fileTexts),
			'text' : fileTexts
			})
	tbert.to_csv(os.path.join(BASEPATH, DATA_DIR, cstr), sep='\t', index=False, header=False)

getTSV(TEST_DIR, 'test.tsv')
getTSV(TRAIN_DIR, 'train.tsv')