import torch
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss

from tqdm import tqdm_notebook, trange
import os
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from transformers.optimization import AdamW, WarmupLinearSchedule

from multiprocessing import Pool, cpu_count
from tools import *
from dataToFeatures import convert_example_to_feature, BinaryClassificationProcessor
from dataProcessing import BASEPATH, DATA_DIR

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bert pre-trained model selected in the list: bert-base-uncased, 
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'bert-base-cased'

# The name of the task to train.I'm going to name this 'yelp'.
TASK_NAME = 'imdb'

# The output directory where the fine-tuned model and checkpoints will be written.
OUTPUT_DIR = 'outputs/imdb/'

# The directory where the evaluation reports will be written to.
REPORTS_DIR = 'reports/imdb_evaluation_report/'

# This is where BERT will look for pre-trained models to load parameters from.
CACHE_DIR = 'cache/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.

MAX_SEQ_LENGTH = 128

TRAIN_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 1
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

output_mode = OUTPUT_MODE

cache_dir = CACHE_DIR

if os.path.exists(REPORTS_DIR) and os.listdir(REPORTS_DIR):
		REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
		os.makedirs(REPORTS_DIR)
if not os.path.exists(REPORTS_DIR):
	os.makedirs(REPORTS_DIR)
	REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
	os.makedirs(REPORTS_DIR)

if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
	raise ValueError("Output directory ({}) already exists and is not empty.".format(OUTPUT_DIR))
if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)

processor = BinaryClassificationProcessor()
train_examples = processor.get_train_examples(DATA_DIR)
train_examples_len = len(train_examples)
label_list = processor.get_labels() # [0, 1] for binary classification
num_labels = len(label_list)
num_train_optimization_steps = int(
	train_examples_len / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
label_map = {label: i for i, label in enumerate(label_list)}
train_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in train_examples]

process_count = cpu_count() - 1
# with open(os.path.join(BASEPATH, DATA_DIR, "train_features.pkl"), "wb") as f:
# 	pass
# if __name__ ==  '__main__':
# 	print(f'Preparing to convert {train_examples_len} examples..')
# 	print(f'Spawning {process_count} processes..')
# 	with Pool(process_count) as p:
# 		train_features = list(tqdm_notebook(p.imap(convert_example_to_feature, train_examples_for_processing), total=train_examples_len))

# with open(os.path.join(BASEPATH, DATA_DIR, "train_features.pkl"), "wb") as f:
# 	pickle.dump(train_features, f)

with open(os.path.join(BASEPATH, DATA_DIR, "train_features.pkl"), "rb") as f:
	train_features = pickle.load(f)
print("hello")
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=num_labels)
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
	{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
	{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]
optimizer = AdamW(optimizer_grouped_parameters,
					 lr=LEARNING_RATE)
					 # warmup=WARMUP_PROPORTION,
					 # t_total=num_train_optimization_steps)
global_step = 0
nb_tr_steps = 0
tr_loss = 0

logger.info("***** Running training *****")
logger.info("  Num examples = %d", train_examples_len)
logger.info("  Batch size = %d", TRAIN_BATCH_SIZE)
logger.info("  Num steps = %d", num_train_optimization_steps)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

if OUTPUT_MODE == "classification":
	all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
elif OUTPUT_MODE == "regression":
	all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
train_sampler = RandomSampler(train_data)
model.train()
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)
# If we save using the predefined names, we can load using `from_pretrained`
output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)
for xiter in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
	tr_loss = 0
	nb_tr_examples, nb_tr_steps = 0, 0
	for step, batch in enumerate(tqdm_notebook(train_dataloader, desc="Iteration")):
		batch = tuple(t.to(device) for t in batch)
		input_ids, input_mask, segment_ids, label_ids = batch

		logits = model(input_ids, segment_ids, input_mask, labels=None)
		logits = logits[0]
		if OUTPUT_MODE == "classification":
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
		elif OUTPUT_MODE == "regression":
			loss_fct = MSELoss()
			loss = loss_fct(logits.view(-1), label_ids.view(-1))

		if GRADIENT_ACCUMULATION_STEPS > 1:
			loss = loss / GRADIENT_ACCUMULATION_STEPS

		loss.backward()
		print("\r%f" % loss, end='')
		
		tr_loss += loss.item()
		nb_tr_examples += input_ids.size(0)
		nb_tr_steps += 1
		if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
			optimizer.step()
			optimizer.zero_grad()
			global_step += 1
	
	if xiter % 10 == 0:
		model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

		torch.save(model_to_save.state_dict(), output_model_file)
		model_to_save.config.to_json_file(output_config_file)
		tokenizer.save_vocabulary(OUTPUT_DIR)

