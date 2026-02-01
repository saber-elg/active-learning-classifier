active-learning-classifier/
├── .github/
│   └── workflows/
│       └── ci.yml                     # CI/CD pipeline configuration
│
├── docs/
│   ├── API.md                         # Complete API reference
│   ├── ARCHITECTURE.md                # System architecture documentation
│   ├── CONTRIBUTING.md                # Contribution guidelines
│   └── DOCKER.md                      # Docker usage documentation
│
├── notebooks/
│   └── no_al_benchmark.ipynb          # Benchmark experiments
│
├── scripts/
│   ├── benchmark_comparison.py        # AL vs Random sampling comparison
│   └── start.sh                       # Quick start script (Unix)
│
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── active_learning.py             # Query strategies implementation
│   ├── config.py                      # Centralized configuration
│   ├── data_preprocessing.py          # Data loading and augmentation
│   ├── model.py                       # CNN architecture
│   └── utils.py                       # Utility functions
│
├── tests/
│   ├── __init__.py
│   └── test_basic.py                  # Unit tests
│
├── .gitignore                         # Git ignore patterns
├── app.py                             # Main Streamlit application
├── CHANGELOG.md                       # Version history
├── docker-compose.yml                 # Docker Compose configuration
├── Dockerfile                         # Docker image definition
├── LICENSE                            # MIT License
├── Makefile                           # Development automation
├── README.md                          # Main project documentation
├── requirements.txt                   # Production dependencies
├── requirements-dev.txt               # Development dependencies
└── setup.py                           # Package installation configuration
