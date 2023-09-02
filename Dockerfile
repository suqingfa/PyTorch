FROM pytorch/pytorch

RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

#RUN echo '' | python