# FlashHead API on native Linux

This guide uses **only** `docker/docker-compose-linux.yml`. It starts the
container with checkpoints mounted from the host, then loads the model through
the API.

Run every command from the repository root.

## 1. Set the checkpoint directory

`CHECKPOINTS_DIR` must be the **host directory that contains both** model
directories:

```text
/data/mehdi/models/
├── SoulX-FlashHead-1_3B/
└── wav2vec2-base-960h/
```

Set it before starting Docker:

```bash
export CHECKPOINTS_DIR=/data/mehdi/models
```

Docker mounts this directory read-only at `/checkpoints` inside the container.
Therefore the two model paths used by the API are:

```text
/checkpoints/SoulX-FlashHead-1_3B
/checkpoints/wav2vec2-base-960h
```

## 2. Optional: choose the host port

The API always listens on port `8000` inside the container. `HOST_PORT` changes
only the port exposed on the Linux host. It defaults to `8000`.

For example, use port `8100` when another service already uses port `8000`:

```bash
export HOST_PORT=8100
```

If port `8000` is free, omit this variable.

## 3. Build, recreate, and start

This is the full command to build the image, recreate the container, and start
it in the background:

```bash
docker compose -f docker/docker-compose-linux.yml up -d --build --force-recreate
```

The `--force-recreate` flag ensures Docker replaces the existing
`flashhead-api` container with one using the current checkpoint mount and port
settings.

## 4. Verify the checkpoint mount

Check that the service is running and that Docker can see the model folders:

```bash
docker ps --filter name=flashhead-api
docker exec flashhead-api ls /checkpoints
```

Expected model folders:

```text
SoulX-FlashHead-1_3B
wav2vec2-base-960h
```

## 5. Load the model

The container starts with `AUTOLOAD=0`, so the model is **not** loaded until you
call `/model/load`.

Set the API URL once. It automatically uses port `8000` if `HOST_PORT` was not
set:

```bash
export API_URL="http://localhost:${HOST_PORT:-8000}"
```

Load the lite model from the mounted checkpoint paths:

```bash
curl -X POST "$API_URL/model/load" \
  -F "ckpt_dir=/checkpoints/SoulX-FlashHead-1_3B" \
  -F "wav2vec_dir=/checkpoints/wav2vec2-base-960h" \
  -F "model_type=lite"
```

Expected response:

```json
{"status":"loaded","model_type":"lite"}
```

## 6. Test that it is ready

```bash
curl "$API_URL/model/status"
```

Expected response:

```json
{"loaded":true,"model_type":"lite","ckpt_dir":"/checkpoints/SoulX-FlashHead-1_3B","wav2vec_dir":"/checkpoints/wav2vec2-base-960h"}
```

The interactive API documentation is at:

```text
$API_URL/docs
```

## Complete copy-paste setup

For a service published on port `8100`:

```bash
export CHECKPOINTS_DIR=/data/mehdi/models
export HOST_PORT=8100
export API_URL="http://localhost:${HOST_PORT}"

docker compose -f docker/docker-compose-linux.yml up -d --build --force-recreate

docker exec flashhead-api ls /checkpoints

curl -X POST "$API_URL/model/load" \
  -F "ckpt_dir=/checkpoints/SoulX-FlashHead-1_3B" \
  -F "wav2vec_dir=/checkpoints/wav2vec2-base-960h" \
  -F "model_type=lite"

curl "$API_URL/model/status"
```

## Useful commands

```bash
# View service logs
docker compose -f docker/docker-compose-linux.yml logs -f

# Stop and remove the container
docker compose -f docker/docker-compose-linux.yml down

# Rebuild, recreate, and start after changing code or Docker files
docker compose -f docker/docker-compose-linux.yml up -d --build --force-recreate
```

## Requirements and troubleshooting

- Native Linux host with an NVIDIA GPU and about 12 GB free VRAM for the lite
  model.
- NVIDIA driver 525 or newer (`nvidia-smi`).
- Docker Compose and NVIDIA Container Toolkit. Verify GPU passthrough with:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
  ```

- If `/model/status` returns `{"loaded":false}`, repeat the `/model/load`
  command above.
- If the checkpoint folders are missing, verify `CHECKPOINTS_DIR` points to
  their parent directory, then rerun the `up -d --build --force-recreate`
  command.
