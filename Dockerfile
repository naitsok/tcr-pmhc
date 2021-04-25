FROM continuumio/miniconda3

WORKDIR /home/biolib

RUN conda install --yes scikit-learn pandas numpy matplotlib lightgbm \
    && \
    conda clean -afy

COPY . .

ENTRYPOINT [ "python", "src/predict_al.py" ]
