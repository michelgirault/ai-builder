FROM ubuntu:jammy

#setup declaration
ENV DEBIAN_FRONTEND=noninteractive

ENV INSTALL_APT="apt install -y"
ENV INSTALL_PIP="python3 -m pip --no-cache-dir install --upgrade --use-deprecated=legacy-resolver --timeout=100" 
ENV PUPPETEER_SKIP_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser
ENV UV_PRERELEASE=allow 

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
    nano \
    git
#install python packages

RUN ${INSTALL_APT} \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    make \
    g++

    # needed 
RUN ${INSTALL_APT} \
    build-essential \
    libcairo2-dev \
    libxt-dev \
    libcogl-pango-dev \
    chromium-browser 

#set directory
WORKDIR /app
COPY . /app

#addon
RUN ${INSTALL_PIP} \
    s3-loader \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    requests \
    #sentence-transformers \
    awscli \
    psycopg \
    psycopg_binary \
    psycopg2-binary \
    Click \
    bcrypt\
    lancedb

#install chroma
RUN ${INSTALL_PIP} chromadb

#install langflow packages
RUN ${INSTALL_PIP} langflow

RUN chmod +x /app/starter.sh
EXPOSE 8080
CMD ["bash", "starter.sh"]