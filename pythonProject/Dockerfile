FROM python:3.9

WORKDIR /pythonProject

COPY requirements.txt /pythonProject/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /pythonProject

EXPOSE 8081

CMD ["python", "main.py","-e", "production"]
ENTRYPOINT ["python", "main.py", "-e", "production"]
