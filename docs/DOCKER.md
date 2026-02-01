# Active Learning Image Classifier

## Quick Reference
- **Maintainer**: Your Name
- **Where to get help**: [GitHub Issues](https://github.com/saber-elg/active-learning-classifier/issues)
- **Where to file issues**: [GitHub Issues](https://github.com/saber-elg/active-learning-classifier/issues)
- **Supported architectures**: amd64, arm64

## What is Active Learning Classifier?

A production-ready active learning framework for efficient image classification with minimal labeled data. This Docker image provides a complete, containerized solution for running the interactive Streamlit application.

## How to use this image

### Quick Start

```bash
docker run -p 8501:8501 active-learning-classifier:latest
```

Then open your browser to http://localhost:8501

### Using Docker Compose

```bash
docker-compose up -d
```

### Persistent Data

Mount volumes to persist your data and models:

```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  active-learning-classifier:latest
```

### Environment Variables

- `STREAMLIT_SERVER_PORT` - Port for Streamlit server (default: 8501)
- `STREAMLIT_SERVER_ADDRESS` - Server address (default: 0.0.0.0)
- `STREAMLIT_SERVER_HEADLESS` - Run in headless mode (default: true)

### Building the Image

```bash
docker build -t active-learning-classifier:latest .
```

### Development Mode

Run with live code reload:

```bash
docker run -p 8501:8501 \
  -v $(pwd):/app \
  active-learning-classifier:latest
```

## Image Variants

### `active-learning-classifier:latest`
This is the standard production image with all dependencies.

### `active-learning-classifier:<version>`
Tagged releases for specific versions.

## License

MIT License - see [LICENSE](https://github.com/saber-elg/active-learning-classifier/blob/main/LICENSE)

## Support

For issues and questions:
- GitHub Issues: https://github.com/saber-elg/active-learning-classifier/issues
- Documentation: https://github.com/saber-elg/active-learning-classifier/blob/main/README.md
