<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>




<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">VizWiz VQA</h3>

  <p align="center">
    Joint Embedding with Transformers
  </p>
</div>



<!-- ABOUT THE PROJECT -->
## Overview & Files

All of the code files in this directory were ran in Kaggle. The code is based on the article written by Tezan Sahu [https://medium.com/data-science-at-microsoft/visual-question-answering-with-multimodal-transformers-d4f57950c867] and further modified to perform our experiments.

There are 1 files included in this folder:
* `vizwiz-models.ipynb`
* `vizwiz-kgs.ipynb`
<br>


### vizwiz-vqa-example.ipynb

This file contains the kaggle notebook about joint embedding with transformers basic model, choice embedding model, and label smoothing functionality with focal loss. 
https://www.kaggle.com/code/lhanhsin/vizwiz-vqa-example

<br>

## Usage

To be able to run these files locally, you'll need to download all the [VizWiz Dataset files](https://vizwiz.org/tasks-and-datasets/vqa/). For each file you will need to adjust the input paths manually to find the `train.json`, `val.json` and the `train/` images from the original VizWiz dataset. Please note that the answer embeddings through RoBERTa for the choice embedding model must be generated manually first.

```
# Sample code to generate answer embeddings
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

word_dl = DataLoader(answer_space, batch_size=10, shuffle=False)
model.to(device)
answer_embedding = []
for i,word_list in enumerate(word_dl):
    encoded_input = tokenizer(word_list,padding=True, return_tensors='pt').to(device)
    output = model(**encoded_input)
    answer_embedding.append(output["pooler_output"])
answer_tensor = torch.cat(answer_embedding,dim=0)
torch.save(answer_tensor, 'answer_embed.pt')
```