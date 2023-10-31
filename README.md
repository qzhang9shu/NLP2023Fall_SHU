## USAGE
`python3 main.py --type="train" --pre2train="True/False"` to train

`python3 main.py --type="predict"` to predict

`python3 main.py --type"generate" --sentence="INPUT HERE"` to generate a summary from the given sentences(1 or more)

## FILE STRUCTURE
```
.
├── checkpoint
├── data
├── evaluate.py
├── generate.py
├── lib
│   ├── criterion.py
│   ├── __init__.py
│   ├── loss.py
│   └── optimizer.py
├── log
├── main.py
├── model
│   ├── attention.py
│   ├── decoder.py
│   ├── embedding.py
│   ├── encoder.py
│   ├── generator.py
│   ├── position_wise_feedforward.py
│   ├── sublayer.py
│   └── transformer.py
├── mT5
│   ├── arg.py
│   ├── data.py
│   ├── run_summarization_mt5.py
│   ├── tools.py
│   └── train_model_summarization.py
├── parser.py
├── prepare_data.py
├── README.md
├── run_summarization_mt5.sh
├── summ_mt5_results
├── train.py
└── utils.py

7 directories, 26 files
```

You can modify the project in parser.py

## Pretrained mode
You can use Hugging Face pretrained model to process the target, but make sure there are enough GPU memory.
run `bash run_summarization_mT5.sh` to make it, and modify in `mT5/arg.py` to fine tune.

## REFERENCE
[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)(一定要看这篇！！！可以直接看英文/上网搜翻译或者机翻)

[Transformer --郑之杰](https://0809zheng.github.io/2020/04/25/transformer.html)

[Transformer | PLM's](https://plmsmile.github.io/2018/08/29/48-attention-is-all-you-need/)

[Transformer pytorch实现逐行详解](https://mdnice.com/writing/fc0b920d4ca84837a5712df1a46865d2)

[从语言模型到Seq2Seq：Transformer如戏，全靠Mask](https://spaces.ac.cn/archives/6933)
