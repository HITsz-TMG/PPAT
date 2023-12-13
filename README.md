# PPAT: Progressive Graph Pairwise Attention Network for Event Causality Identification

## Setup

```
conda create -n PPAT python=3.8
conda activate PPAT
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

Download BERT-base from [here](https://huggingface.co/bert-base-uncased) and put it under **encoder/BERT-base/** folder.
Editing the **vocab.txt** as following (adding &lt;t&gt; and &lt;/t&gt; as event marker tokens):
```
[PAD]
<t>
</t>
[unused2]
[unused3]
[unused4]
[unused5]
...
```

## Data

We use EventStoryLine(v0.9) and Causal-TimeBank. The data is in the ''data'' folder.
The MAVEN-ERE data set can be found in [here](https://github.com/THU-KEG/MAVEN-ERE)

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
