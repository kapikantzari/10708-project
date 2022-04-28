# 10708 Project: Improving Semantic Relation Prediction using Global Graph Properties
Group Members: Yun Cheng, Yuxin Pei, Eric Liang

The implementation of this code is based off of the M3GM model created by Yuval Pinter. The original paper can be found [here](https://arxiv.org/abs/1808.08644). In addition the code along with details of the original model and how to run the original model can be found [here](https://github.com/yuvalpinter/m3gm).

In order to reproduce our results and run our experiments we provide a summary below. Please also note that because of the time and computing power it takes to run this code it is reccomended to use some sort of computing service to run this code. Also not for convenience of use we have included a pre-trained model that can be donwloaded [here](https://drive.google.com/drive/folders/1VUCk5Tj5yhWce_geeu-I-XjunHSg8Utz?usp=sharing) (the pre-trained model file could not be included in our github repository due to the size of the model). This pre-trained model was for the transE association operator as described in our report.

## How to train and predict the model
We based our model's structure off of the original M3GM model, so for easy use for new users we kept many of the command line argument flags. For in depth details of each command flag, please see the "Association Model" and "Max-Margin Markov Graph Models" sections [here](https://github.com/yuvalpinter/m3gm). We provide a brief summary on how to get our modified model running below.

In order to train the association model an example of the prompt to run the code is seen below. Note that the code to train the association model is under pretrain_assoc.py, however please defer to below sections to see how to run modified experiments as described in our report.
```
python pretrain_assoc.py --input data/wn18rr.pkl --embeddings data/ft-embs-all-lower.vec --model-out models/pret_transE --nll --assoc-mode transE --neg-samp 10 --early-stopping --eval-dev
```

In order to train the max-margin markov graph model and do prediction an example of the prompt to run the code is seen below. Note that the code to train and do prediction of the model is under predict_wn18.py, but in order t run modified experiments for sampling, etc. as described in our report see below sections.
```
python predict_wn18.py --input data/wn18rr.pkl --emb-size 300 --model models/pret_transE-ep-14 --model-only-init --assoc-mode transE --eval-dev --no-assoc-bp --epochs 3 --neg-samp 10 --regularize 0.01 --rand-all --skip-symmetrics --model-out models/from_pret_trE-3eps --rerank-out from_pret_trE-3eps.txt
```

## How to run sampling experiments
In order to run the sampling experiments, the changes are housed in training.py. To run the three expriments with negative sampling, importance sampling, and hierarchical softmax, please refer to the instructions in lines 23-27 of training.py, in summary new users just need to modify the "samplingType" variable to be either "negative","importance", or "hierarchical" in order to easily swap between the experiments. Further implementation details and comments can be seen in this file.
