## With Docker

Entry point python
```bash
docker run -it --rm python:3.8.12-slim
```


Default entry point changed from python to bash

```bash
docker run -it --rm  --entrypoint=bash python:3.8.12-slim
```