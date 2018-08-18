FROM centos:7

RUN yum -y install epel-release
RUN yum -y update

RUN yum install -y python-pip
RUN pip install --upgrade pip

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["api.py"]
