# Res Publica: automated analysis of Italian political texts

**Res Publica** is a simple machine learning model applied to texts of political manifestos, annotated by the political scientists of the [Manifesto Project](https://manifestoproject.wzb.eu/). 

The idea is to use the high-quality (but relatively low volume) manifesto project data annotated by human experts in order to train a text-classification model that can be used to extrapolate the experts' annotations to larger text corpora such as news articles. The hope is to support political education. 

This code is based on a couple of earlier projects, namely:

* [fipi (fuer ihre politische information)](https://github.com/felixbiessmann/fipi);
* [political-affiliation-prediction](https://github.com/kirel/political-affiliation-prediction), which learned a similar text classification model on speeches in the German Parliament.


## Installation

### Local setup in virtualenv

Install [virtualenvwrapper](https://virtualenvwrapper.readthedocs.org/en/latest/). 
In the folder containing the directory cloned from github then type:

    mkvirtualenv -a respublica respublica

Go to the `respublica` folder and install the dependencies with

    pip install -r requirements.txt

Start the webserver with 
    
    python api.py

Open a browser window and navigate to localhost:5000. 

### Local setup with Docker

Install [Docker](https://docs.docker.com/engine/installation/) and start it. 
In the project root folder then build the docker image and start it with:

    docker-compose up

Open a browser window and navigate to `[IP-of-docker-container]:5000`.
Note: be aware of bug [#14755](https://github.com/moby/moby/issues/14755).

### Deploy with AWS Elasticbeanstalk

Install EB CLI
    
    pip install awsebcli

Create and deploy app, then open it

    eb init
    eb create
    eb open
