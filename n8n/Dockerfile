FROM ubuntu:jammy

#setup declaration
ENV DEBIAN_FRONTEND=noninteractive

ENV INSTALL_APT="apt install -y"
ENV INSTALL_PIP="python3 -m pip --no-cache-dir install --upgrade"
ENV N8N_USER_FOLDER=.n8n

#create user for this container
RUN adduser --uid 1999 n8n


#update packages
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
	build-essential

#install git and lfs
RUN ${INSTALL_APT} \
 	git \
	git-lfs \
	wget

#install python packages
RUN ${INSTALL_APT} \
	python3 \
	python3-pip \
	python3-dev \
	make \
	g++
#install pip packages
RUN ${INSTALL_PIP} \
	huggingface_hub \
	python-magic

#install nodejs
RUN curl -s https://deb.nodesource.com/setup_18.x | bash
RUN apt install -y nodejs
#set working directory

WORKDIR /app
COPY . /app

#install n8n
RUN set -eux; \
	npm install -g n8n
RUN chmod +x /app/starter.sh
#RUN mkdir .n8n 
RUN chown -R 1999:1999 /app 
#switch to the user of the container
USER n8n
#open port for use
EXPOSE 5678
CMD ["bash", "starter.sh"]