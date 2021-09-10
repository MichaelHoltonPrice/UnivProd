FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get install -y python3-setuptools && \
    apt-get install -y ipython3 && \
    apt-get install -y vim && \
    apt-get install -y git && \
    apt-get install -y libicu-dev && \
    apt-get clean

# Make a data directory that will be mirrored to the host
RUN mkdir /data

# copy the dependencies file to the working directory (the root, /)
COPY requirements.txt .

# Copy script files
COPY parse_Papers.py .

# install dependencies
RUN pip3 install -r requirements.txt

# docker build -t michaelholtonprice/complementarity .
# docker run --name mag -itv //c/Users/mpatm/localMAG:/data michaelholtonprice/complementarity
