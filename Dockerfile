FROM ubuntu:20.04

    ########################
    # Install apt packages #
    ########################

# Upgrade and install packages:
#
# ffmpeg: Needed for animations in examples.
# gcc: C compiler. Needed for building viridicle.
# libfreetype6-dev: Needed for matplotlib in examples.
# python3-dev: Python with development headers.
# python3-pip: Python package manager

RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ffmpeg \
  gcc \
  libfreetype6-dev \
  python3-dev \
  python3-pip

    #####################
    # Install viridicle #
    #####################

COPY . /opt/viridicle
WORKDIR /opt/viridicle
RUN python3 -m pip install .
RUN python3 -m pip install -r requirements-examples.txt
