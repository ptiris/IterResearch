# IterResearch Python Sandbox

This directory contains a minimal local Python sandbox service compatible with `tools/python_interpreter.py`.

## API

### `POST /run`

Request body:

```json
{
  "code": "print('hello')",
  "language": "python",
  "run_timeout": 50
}
```

Response body:

```json
{
  "stdout": "hello\n",
  "stderr": "",
  "execution_time": 0.123456
}
```

## Run with Docker

```bash
cd sandbox
docker compose up --build
```

## Configure IterResearch

Set this in your `.env`:

```bash
export SANDBOX_ENDPOINTS="http://127.0.0.1:8080"
```

## Notes

- The service executes code inside the container, which provides the main isolation boundary.
- For stronger isolation, run the container with additional runtime restrictions such as CPU/memory limits and a read-only filesystem.
