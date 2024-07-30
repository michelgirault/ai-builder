FROM ubuntu:jammy

#setup declaration
ENV DEBIAN_FRONTEND=noninteractive

ENV INSTALL_APT="apt install -y"
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

# needed 
RUN ${INSTALL_APT} \
    build-essential \
    libcairo2-dev \
    libxt-dev \
    libcogl-pango-dev \
    chromium-browser 

#install node repo
RUN curl -s https://deb.nodesource.com/setup_18.x | bash
RUN apt install nodejs -y

#addon
RUN ${INSTALL_PIP} \
    chromadb \
    llama-index
# install node red
RUN npm install -g --unsafe-perm node-red

WORKDIR /app
EXPOSE 1880
CMD ["node-red"]