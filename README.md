# PPAT: Progressive Graph Pairwise Attention Network for Event Causality Identification

## Setup

```
conda create -n PPAT python=3.8
conda activate PPAT
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

Download BERT-base from [here](https://huggingface.co/bert-base-uncased) and put it under "encoder/BERT-base/" folder.

## Data

We use EventStoryLine(v0.9) and Causal-TimeBank. The data is in the ''data'' folder.

## Running

running cross-validation on EventStoryLine
```
cd PPAT
python PPAT_framework.py --device {GPU device} --run_mode stack5
```

running one fold on EventStoryLine
```
cd PPAT
python PPAT_framework.py --device {GPU device} --run_mode train0
```

running one fold on Causal-TimeBank
```
cd PPAT
python PPAT_framework.py --device {GPU device} --run_mode train0 --stack_datapath "../data/ctb_stack10_123"  --dev_datapatg "../data/ctb_stack10_123/0/test.json"
```
