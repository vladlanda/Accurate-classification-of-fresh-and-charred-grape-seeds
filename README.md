# Accurate-classification-of-fresh-and-charred-grape-seeds

#### The application works with to types ot files : 

 1. WRL 
 2. MAT (wich contains edges and vertices)

#### Use *environment.yml* to create conda enviroment.

#### Run train & classify:
All the training and testing files should be plased in **data_set** folder, as follows:
```markdown
├── data_set
│   ├── test
│   │   ├── 1.wrl or .mat
│   │   ├── 2.wrl or .mat
│   │   :
│   │   ├── n.wrl or .mat
│   ├── train
│   │   ├── class1
│	│   │   ├── 1.wrl or .mat
│	│   │   ├── 2.wrl or .mat
│	│   │   :
│	│   │   ├── n.wrl or .mat
│   │   ├── class2
│	│   │   ├── 1.wrl or .mat
│	│   │   ├── 2.wrl or .mat
│	│   │   :
│	│   │   ├── n.wrl or .mat
		:
│   │   ├── class n
│	│   │   ├── 1.wrl or .mat
│	│   │   ├── 2.wrl or .mat
│	│   │   :
│	│   │   ├── n.wrl or .mat
```

Based on the files, run : **run_app.bat** or **run_app_mat_files.bat** (check the .sh for the run commad).
The **run** options are:
```markdown
usage: app.py [-h] [-d D] [-m M] [-r R] [-t T] [-ni NI] [-np NP] [-nb NB] [-debug] [-random] [-sts] [-mat] [-minfiles MINFILES]
              [--ratios RATIOS [RATIOS ...]]

LDA+ICP base seeds clasifiier, Example : python app.py -np 2500 -nb 4 -random -ni 1

optional arguments:
  -h, --help            show this help message and exit
  -d D                  Data sets folder containing test and train folders (default "-d ./data_set")
  -m M                  Models folder to save and load models from (default "-m ./models")
  -r R                  Result folder to save results and figures (default "-m ./results")
  -t T                  Classifier type : (tour) Tournament trainer or (simple) Simple trainer
  -ni NI                Number of iterations (default 100)
  -np NP                Number of cloud points (default 2500)
  -nb NB                Number of classes in each model (default 4)
  -debug                Enable debug mode
  -random               Enable random shuffle of models while training
  -sts                  Check current state dict and number of iterations
  -mat                  Files in the train and test directories are .mat files (matlab)
  -minfiles MINFILES    Minimum files in train folder, otherwise skip folder
  --ratios RATIOS [RATIOS ...]
                        The ratios range used in order to filter Archiological samples (--ratios <low> <high>)
```