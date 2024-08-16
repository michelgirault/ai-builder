FROM ubuntu:jammy

#setup declaration
ENV DEBIAN_FRONTEND=noninteractive

ENV INSTALL_APT="apt install -y"
ENV INSTALL_PIP="python3 -m pip --no-cache-dir install --upgrade"
ENV PUPPETEER_SKIP_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser

#start

USER root
RUN apt -y update && apt -y upgrade
#install dependencies
RUN ${INSTALL_APT} \
    libz-dev \
    libssl-dev \
    libcurl4-gnutls-dev \
    libexpat1-dev \
    gettext \
    cmake \
    gcc \
    git
#install python packages

RUN ${INSTALL_APT} \
    python3 \
    python3-pip \
    python3-dev \
    make \
    g++

    # needed 
RUN ${INSTALL_APT} \
    build-essential \
    libcairo2-dev \
    libxt-dev \
    libcogl-pango-dev \
    chromium-browser 

#addon
RUN ${INSTALL_PIP} \
    chromadb \
    s3-loader \
    pip install boto3

#set directory
WORKDIR /app
COPY . /app

#install langflow packages
RUN ${INSTALL_PIP} langflow

RUN chmod +x /app/starter.sh
EXPOSE 8080
CMD ["bash", "starter.sh"]