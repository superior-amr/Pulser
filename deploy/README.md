# Oracle Cloud VPS Deployment

## Prerequisites
- Oracle Cloud free tier account (no credit card): https://cloud.oracle.com
- SSH key pair (generated during VM creation)

## Step 1: Create VM

1. Go to Compute → Instances → Create Instance
2. Choose:
   - **Image**: Ubuntu 22.04 or Oracle Linux 9
   - **Shape**: VM.Standard.A1.Flex (Always Free)
   - **OCPU**: 4, **RAM**: 24 GB
   - **Boot volume**: 50 GB
3. Upload or generate SSH key
4. Create and copy the **Public IP**

## Step 2: Connect

```bash
ssh -i ~/Downloads/your-key.pem ubuntu@YOUR_PUBLIC_IP
```

## Step 3: Run Setup

```bash
curl -sL https://raw.githubusercontent.com/superior-amr/Pulser/main/deploy/setup.sh | bash
```

Or manually:

```bash
git clone https://github.com/superior-amr/Pulser.git
cd Pulser
bash deploy/setup.sh
```

## Step 4: Open Firewall

In Oracle Cloud console:
1. Go to your VM → Virtual Cloud Network → Security Lists
2. Add Ingress Rule:
   - Source: `0.0.0.0/0`
   - Destination Port: `8000`

## Step 5: Access

Open `http://YOUR_PUBLIC_IP:8000` in your browser.

## Service Commands

```bash
sudo systemctl status pulser       # check status
sudo systemctl restart pulser      # restart
sudo systemctl stop pulser         # stop
sudo journalctl -u pulser -f       # view live logs
```
