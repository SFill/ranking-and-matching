FROM ubuntu:18.04 

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \ 
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install conda-build && \
     /opt/conda/bin/conda clean -ya 
ENV PATH /opt/conda/bin:$PATH

RUN conda install numpy==1.19.2 pyyaml==5.4.1 scipy==1.6.1 nltk==3.5 pandas==1.2.3 && /opt/conda/bin/conda clean -ya
RUN conda install pytorch==1.7.1 cpuonly -c pytorch && /opt/conda/bin/conda clean -ya
RUN conda install scikit-learn==0.23.2 faiss-cpu==1.7.0 -c conda-forge && /opt/conda/bin/conda clean -ya

RUN python -m nltk.downloader punkt
RUN python -m pip install Flask==2.2.2 langdetect==1.0.8

WORKDIR /workspace
COPY additional_data/ /additional_data/

RUN chmod -R a+w /workspace

COPY . .

ENTRYPOINT [ "bash", "-c", "FLASK_APP=main.py FLASK_DEBUG=1 WERKZEUG_DEBUG_PIN=off flask run --port 11000 --host=0.0.0.0" ]