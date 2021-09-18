## Setting up environment

*This section is explained assuming the Operating System is a Linux Distribution*

### Check python version

~~~
python --version
~~~

### PIP (Package Installer for Python)

Check if PIP is already installed.
~~~
pip --version
~~~ 

In case it is not installed, update repository and install it.
~~~
sudo apt-get update \
sudo apt-get install python3-pip
~~~

### Create virtual environment

First install `virtualenv` package.
~~~
pip install virtualenv
~~~

Clone Github repository in your desired directory.
~~~
git clone https://github.com/estebanfw/automatic-collision-detection.git
~~~

Create virtual environment.
~~~
python -m venv venv
~~~
Activate virtual environment.
~~~
source venv/bin/activate
~~~
Install required libraries.
~~~
pip install -r requirements.txt
~~~
To list all the installed packages in the virtual environment: `pip freeze`

## Files structure

* `create_dataframe.py` makes all the ETL process to get a clean dataset for building the machine learning module. Output dataset of this script consist only on risky events (collision probabilities greater than 10^-6)
* `preparing_data.py` contains all the functions needed in the above script to do the ETL.
* `full_range.py`makes all the ETL to build a dataset with colision probabilities in al the range.
* `ml_module.py` contains all the functions needed for building and/or loading the Light GBM model.
* `bayesian opt` python notebook that uses as input the clean dataset and the functions of the `ml_module` to build the Machine Learning Model and view the results.
