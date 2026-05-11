# Deploying This Flask Project

This project uses older TensorFlow and Keras versions, so Docker deployment is the most reliable path.

## Run Locally With Docker

```bash
docker build -t ad-click-fraud-detection .
docker run -p 10000:10000 ad-click-fraud-detection
```

Open:

```text
http://localhost:10000
```

## Deploy On Render With Docker

1. Push this project folder to GitHub.
2. Create a new Render Web Service.
3. Select Docker as the runtime.
4. Keep the default Dockerfile path: `Dockerfile`.
5. Set the port to `10000` if Render asks for it.

The app reads the `PORT` environment variable automatically.

## Login

```text
Username: chetan
Password: 12345
```
