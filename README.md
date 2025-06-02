# Mechanistic Explorer

A no-code interactive dashboard for mechanistic interpretability research. Visualize and explore how language models make predictions layer-by-layer using the logit lens technique.

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

Mechanistic Explorer provides researchers and enthusiasts with an intuitive interface to:
- 🔍 Visualize token predictions across all model layers
- 💬 Interact with models through multi-turn conversations
- 📊 Analyze how predictions evolve through the transformer layers
- 🎯 Click on any token to see detailed layer-by-layer predictions

## Features

- **Real-time Logit Lens Visualization**: See how the model's predictions change at each layer
- **Interactive Token Analysis**: Click any token to explore its prediction evolution
- **Multi-turn Conversations**: Maintains context across multiple interactions
- **Two Visualization Modes**: 
  - Standard view: Detailed predictions per layer
  - Heatmap view: Visual representation of probability distributions
- **Local Model Support**: Uses GPT-2 locally via nnsight library

## Prerequisites

- Python 3.8 or higher
- Node.js 16.x or higher
- npm or yarn
- 4GB+ RAM recommended for running GPT-2 locally

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mechanisticexplorer.git
cd mechanisticexplorer
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: First run will download the GPT-2 model (~500MB).

### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install
```

## Running the Application

### 1. Start the Backend Server

```bash
cd backend
python run.py
```

The backend will start at `http://localhost:8000`. You can verify it's running by visiting `http://localhost:8000/api/health`.

### 2. Start the Frontend Development Server

In a new terminal:

```bash
cd frontend
npm run dev
```

The frontend will start at `http://localhost:5173`.

## Usage

1. Open your browser and navigate to `http://localhost:5173`
2. Type a message in the chat input
3. Watch as the model generates a response
4. Click on any token to see how the model's predictions evolved through its layers
5. Use the view toggle to switch between standard and heatmap visualizations

## Architecture

```
mechanisticexplorer/
├── backend/
│   ├── app/
│   │   ├── main.py        # FastAPI application with nnsight integration
│   │   └── models.py      # Pydantic data models
│   ├── requirements.txt   # Python dependencies
│   └── run.py            # Backend server runner
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── context/      # React context for state management
│   │   ├── api/          # API client
│   │   └── types/        # TypeScript type definitions
│   └── package.json      # Node dependencies
└── docs/
    └── usage.md          # Detailed usage guide
```

## How It Works

The logit lens technique allows us to peek into the model's intermediate layers and see what it would predict at each stage. This implementation:

1. **Traces Model Execution**: Uses nnsight to trace through GPT-2's layers
2. **Extracts Predictions**: At each layer, applies the language model head to get token predictions
3. **Visualizes Evolution**: Shows how predictions change from early to late layers

## API Endpoints

- `POST /api/chat` - Send messages and receive responses with logit lens data
- `GET /api/health` - Check API health status
- `GET /api/model_info` - Get information about the loaded model

## Development

### Adding New Models

To use a different model, update the `model_name` in `backend/app/main.py`:

```python
model_name = "your-model-name"  # Any Hugging Face model
```

Note: You may need to adjust the layer extraction logic for different architectures.

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Ensure you have a stable internet connection
   - Check available disk space (need ~1GB for GPT-2)

2. **Backend Won't Start**
   - Verify Python version: `python --version`
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

3. **Frontend Connection Error**
   - Verify backend is running on port 8000
   - Check CORS settings if using different ports

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [nnsight](https://nnsight.net/) for model interpretability
- Inspired by the logit lens research paper
- Uses GPT-2 from OpenAI/Hugging Face

## Contact

For questions or feedback, please open an issue on GitHub.