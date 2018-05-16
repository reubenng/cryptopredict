# CNN

## Instructions

### Preprocessing

Firstly, place all your price csv data found from `bitcoinminutescraper.py`
and place it into `data/prices/` and run

```bash
cd ./data/prices/ && bash ./combine.bash`
```

Do the same thing with tweet csv data found from `twitter2.py` and place it in
`data/tweets` and run `cd ./data/tweets && bash ./combine.bash`.

You can now preprocess the data to combine the tweets with the price data to
create a csv file, `data/made.csv`.
Then, this can be turned into the input/ouput data fron the CNN.
To do both these steps, run:

```bash
python3 CNN.py
```

### Training CNN

Run:

```bash
python3 keras_cnn.py
```

to train the CNN for 100 epochs.

Use:

```bash
tensorboard --logdir logs/
```

and go to http://localhost:6006  (or whatever url tensorboard says to use)
to see the progress of training in a nice graphical way.
