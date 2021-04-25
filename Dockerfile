FROM continuumio/miniconda3

WORKDIR /home/biolib

SHELL ["/bin/bash", "--login", "-c"]

RUN conda create -c conda-forge -n allennlp python=3.7

RUN conda init bash

RUN conda activate allennlp 

RUN conda install --yes pytorch scikit-learn pandas numpy matplotlib \
    && \
    conda clean -afy

COPY . .

ENTRYPOINT [ "python", "src/predict.py" ]
