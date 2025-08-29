# Secure Deployment with Cloudflare Tunnel

This guide provides step-by-step instructions on how to securely publish your API on the internet using [Cloudflare Tunnel](https://www.cloudflare.com/products/tunnel/).

This method is highly secure, as it does not expose your server's IP address directly and encrypts all traffic between Cloudflare and your server.

## Prerequisites

Before you begin, you will need:

1.  **A Cloudflare Account:** You can create one for free at [cloudflare.com](https://www.cloudflare.com/).
2.  **A Domain Name:** You need to own a domain name and have it added to your Cloudflare account.
3.  **`cloudflared` Installed:** You must install the Cloudflare Tunnel command-line tool (`cloudflared`) on the machine where you will run the API.
    -   Follow the official installation instructions here: [Install `cloudflared`](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/)

## Deployment Steps

### Step 1: Authenticate `cloudflared`

First, you need to link the `cloudflared` tool to your Cloudflare account. Run the following command in your terminal:

```bash
cloudflared login
```

This will open a browser window asking you to log in to your Cloudflare account and authorize the tool for the domain you intend to use.

### Step 2: Install Dependencies

Make sure you have installed all the required Python packages, including the new `gunicorn` server:

```bash
pip install -r ollama_chat_rag/requirements.txt
```

### Step 3: Start the API Application

Next, start the API in **production mode**. The following command uses `gunicorn` to run the application and puts it in the background so you can proceed to the next step.

```bash
python -m ollama_chat_rag.cli start --prod --background
```

You should see a message confirming that the application has started in the background. The API is now running locally, listening on `127.0.0.1:8000`.

### Step 4: Launch the Cloudflare Tunnel

Now, create the secure tunnel to expose your local API to the internet. Run the following command, but **make sure to replace `api.your-domain.com` with your desired public hostname**.

```bash
cloudflared tunnel --url http://127.0.0.1:8000 --hostname api.your-domain.com
```

-   `--url http://127.0.0.1:8000`: This tells `cloudflared` to forward incoming traffic to your local API server.
-   `--hostname api.your-domain.com`: This is the public URL where your API will be accessible. **You must change this to your own domain/subdomain.**

`cloudflared` will now connect to the Cloudflare network. Once it's running, your API will be live and accessible at the hostname you specified, secured with HTTPS.

## How to Stop Everything

1.  **Stop the Tunnel:** Press `Ctrl+C` in the terminal where `cloudflared` is running.
2.  **Stop the Application:** You need to find the background process and stop it. You can find the process ID (PID) with a command like `pgrep -f gunicorn` and then stop it with `kill <PID>`.
