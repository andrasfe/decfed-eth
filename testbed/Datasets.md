# Generation of datasets

* Used the [MNIST as .jpg](https://www.kaggle.com/datasets/scolianni/mnistasjpg?select=trainingSet) Kaggle dataset.
* Copied into `testbed/datasets`. 
* Un-tar it
* Run `python3 data_loader.py --no_trainers 10`. This will populate the client and owner datasets. Depending on number of desired clients, change the value accordingly. It will split the dataset into N + 1, including the owner's validation set.

