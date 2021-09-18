## Setting up environment

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

Finally create these folders:
~~~
mkdir data dataframe validation-results
~~~