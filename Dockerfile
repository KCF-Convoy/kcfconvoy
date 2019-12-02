FROM conda/miniconda3

WORKDIR /opt/kcfconvoy
COPY . /opt/kcfconvoy/

ENV PYTHONPATH=/usr/local/lib/python3.8/site-packages
RUN conda install -c conda-forge rdkit
RUN python3 setup.py install

CMD ["tail", "-f", "/dev/null"]
