FROM swipl:latest

# Install Python + toolchain
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    gcc make libgmp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ensure python is python3
RUN ln -s /usr/bin/python3 /usr/bin/python || true

WORKDIR /app

COPY requirements.txt .

# ✅ Allow pip installs inside Debian Bookworm
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001
CMD ["python", "app.py"]
