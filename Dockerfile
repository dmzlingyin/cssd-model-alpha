FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./server.py" ]
