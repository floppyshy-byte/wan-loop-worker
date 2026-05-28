FROM runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel

WORKDIR /

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY rp_handler.py /

CMD ["python3", "-u", "rp_handler.py"]
