FROM conda/miniconda3

WORKDIR /opt/kcfconvoy
COPY . /opt/kcfconvoy/

RUN python3 setup.py install

CMD ["tail", "-f", "/dev/null"]
