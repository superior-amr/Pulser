#!/bin/bash
set -e

echo "=== Pulser RAG — Oracle Cloud Setup ==="

# Detect package manager
if command -v dnf &> /dev/null; then
    PKG_MGR="sudo dnf"
elif command -v apt &> /dev/null; then
    PKG_MGR="sudo apt"
    sudo apt update -y
else
    echo "Unsupported OS. Use Oracle Linux 9 or Ubuntu 22.04."
    exit 1
fi

# Install Python 3.11 and git
echo "[1/6] Installing dependencies..."
if [ "$PKG_MGR" = "sudo dnf" ]; then
    sudo dnf install -y git python3.11 python3.11-pip python3.11-venv
else
    sudo apt install -y git python3.11 python3.11-venv python3-pip
fi

# Clone repo (skip if already cloned)
echo "[2/6] Setting up project..."
if [ ! -d "Pulser" ]; then
    git clone https://github.com/superior-amr/Pulser.git
fi
cd Pulser

# Create venv and install deps
echo "[3/6] Installing Python packages..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set up systemd service
echo "[4/6] Setting up service..."
read -p "Enter your GROQ_API_KEY: " GROQ_KEY

sudo tee /etc/systemd/system/pulser.service > /dev/null <<EOF
[Unit]
Description=Pulser RAG
After=network.target

[Service]
User=$(whoami)
WorkingDirectory=$(pwd)
Environment=GROQ_API_KEY=$GROQ_KEY
ExecStart=$(pwd)/venv/bin/gunicorn app:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
echo "[5/6] Starting service..."
sudo systemctl daemon-reload
sudo systemctl enable pulser
sudo systemctl restart pulser

echo "[6/6] Done!"
echo ""
echo "=== Pulser RAG is running ==="
echo "Public URL: http://$(curl -s ifconfig.me):8000"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status pulser    — check status"
echo "  sudo systemctl restart pulser   — restart"
echo "  sudo journalctl -u pulser -f    — view logs"
