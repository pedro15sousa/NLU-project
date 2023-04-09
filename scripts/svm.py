from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn import svm
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset
from data_preprocessing_2 import label_dict
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import classification_report




# Step 1: Prepare the data

# Load the pre-processed datasets
base_path = Path(__file__).parent
repo_path = (base_path / "../data/processed").resolve()
df = pd.read_csv(repo_path / 'total.csv')

# # Replace all 'val' values with 'test'
# df['data_type'].replace({'val': 'test'}, inplace=True)

# # Split the 'train' data into 'train' and 'test'
# train_df = df[df['data_type'] == 'train']
# train_data, val_data = train_test_split(train_df, test_size=0.5)

# # Update the dataframe accordingly
# df.loc[train_data.index, 'data_type'] = 'train'
# df.loc[val_data.index, 'data_type'] = 'val'

#find the maximum length
max_len = max([len(sent) for sent in df.Tweets])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case = True)

# tokenize test set
encoded_data_train = tokenizer.batch_encode_plus(df[df.data_type == 'train'].Tweets.values,
                                                 return_attention_mask=True,
                                                 pad_to_max_length=True,
                                                 max_length=max_len,
                                                 return_tensors='pt')
                            
# tokenize val set
encoded_data_val = tokenizer.batch_encode_plus(df[df.data_type == 'val'].Tweets.values,
                                                #add_special_tokens = True,
                                                return_attention_mask = True,
                                                pad_to_max_length = True,
                                                max_length = max_len,
                                                return_tensors = 'pt')

# encode train set
input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type == 'train'].label.values)

# encode val set
input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
# convert data type to torch.tensor
labels_val = torch.tensor(df[df.data_type == 'val'].label.values)

#create dataloader
dataset_train = TensorDataset(input_ids_train, 
                              attention_masks_train,
                              labels_train)

dataset_val = TensorDataset(input_ids_val, 
                             attention_masks_val, 
                             labels_val)



#load pre-trained BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels = len(label_dict),
                                                      output_attentions = True,
                                                      output_hidden_states = True)


batch_size = 4 #since we have limited resource

#load train set
dataloader_train = DataLoader(dataset_train,
                              sampler = RandomSampler(dataset_train),
                              batch_size = batch_size)

#load val set
dataloader_val = DataLoader(dataset_val,
                              sampler = RandomSampler(dataset_val),
                              batch_size = 32) #since we don't have to do backpropagation for this step


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
features_train = []
for batch in dataloader_train:
    batch = tuple(b.to(device) for b in batch)
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1]}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states[-2]
    features_train.append(hidden_states.cpu().numpy())

# concatenate list of features into matrix
features_train = np.concatenate(features_train, axis=0)

# reshape to remove batch dimension
features_train = np.reshape(features_train, (features_train.shape[0], -1))


# repeat for val set
features_val = []
for batch in dataloader_val:
    batch = tuple(b.to(device) for b in batch)
    inputs = {'input_ids':      batch[0],
              'attention_mask': batch[1]}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states[-2]
    features_val.append(hidden_states.cpu().numpy())

features_val = np.concatenate(features_val, axis=0)

# reshape to remove batch dimension
features_val = np.reshape(features_val, (features_val.shape[0], -1))


# train SVM model
clf = svm.SVC()
clf.fit(features_train, labels_train)

# predict on validation set
val_preds = clf.predict(features_val)

# print results
print('Validation set predictions:', val_preds)

# generate classification report
target_names = list(label_dict.keys())
svm_report = classification_report(labels_val, val_preds, target_names=target_names)

print('SVM classification report:')
print(svm_report)