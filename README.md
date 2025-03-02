# Logit Lens UI

A visualization tool for exploring language model predictions across different layers. This application provides an interactive interface to analyze how token predictions evolve through the model's layers, helping to understand the model's decision-making process.

## Features

- Interactive heatmap visualization of token predictions across model layers
- Color-coded probability representation
- Token-by-token analysis
- Context token visualization
- Mock API backend for development and testing
- FastAPI backend ready for integration with real model inference

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py        # FastAPI application
│   │   └── models.py      # Data models
│   └── run.py             # Backend server runner
└── frontend/
    ├── src/
    │   ├── components/    # React components
    │   ├── context/       # React context providers
    │   └── api/          # API client
    └── package.json
```

## Setup and Running

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```bash
   python run.py
   ```

The backend server will start at `http://localhost:8000`. You can check the API health at `http://localhost:8000/api/health`.

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

The frontend development server will start at `http://localhost:5173`.

## API Endpoints

- `POST /api/chat`: Send messages and receive token predictions
- `GET /api/health`: Check API health status

## Development

The current implementation uses a mock backend that simulates token predictions. The mock data generation logic in `backend/app/main.py` can be replaced with actual model inference code when needed.

### Mock Data Generation

The backend currently generates mock data that simulates:
- Token predictions across 24 layers
- Alternative token suggestions
- Probability distributions that evolve through layers
- Realistic response generation for specific queries

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License - feel free to use this code for your own projects. 