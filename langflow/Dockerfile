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
#install python

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
    llama-index
#install
RUN python3 -m pip install langflow -U

WORKDIR /app
EXPOSE 7860
CMD ["python3", "-m", "langflow" , "run" , "--host", "0.0.0.0", "--port", "8080"]