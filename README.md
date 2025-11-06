# Fire Evacuation Route Planning System

A real-time fire evacuation route planning system that uses ant colony optimization (ACO) and A* pathfinding to calculate optimal escape routes in buildings during fire emergencies.

## Features

- **Dynamic Fire Simulation**: Models fire spread with three stages (initial, growth, spread) using diffusion and amplification
- **Intelligent Pathfinding**: Combines ant colony optimization with A* algorithm for optimal route calculation
- **Multi-Floor Support**: Handles evacuation planning across multiple building floors
- **Safety-First Routing**: Considers fire intensity, obstacles, and safety buffers when planning paths
- **Turn-by-Turn Navigation**: Generates detailed navigation instructions with distances and turning points
- **Visual Output**: Creates evacuation route visualizations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd live_escape
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the API Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Get Evacuation Route

```
GET /evacuation
```

**Parameters:**
- `start_row` (int): Starting row position
- `start_col` (int): Starting column position
- `strating_floor` (int): Starting floor number (0-2)
- `fire_locations` (list[str]): Fire positions in format "row,col" (can specify multiple)
- `fire_floor` (int): Floor number where fire is located
- `exits` (list[str]): Exit positions in format "row,col" (can specify multiple)
- `stage` (str): Fire stage - "initial", "growth", or "spread" (default: "initial")

**Example Request:**
```
GET /evacuation?start_row=5&start_col=5&strating_floor=0&fire_locations=10,10&fire_floor=0&exits=0,0&exits=20,20&stage=growth
```

**Response:**
```json
{
  "path": [[5,5], [5,6], ...],
  "length": 25.4,
  "turning_points_count": 3,
  "turning_points": [...],
  "navigation_instructions": [...],
  "download_url": "/download/evacuation_route.png",
  "fire_considered": true
}
```

#### Download Route Visualization

```
GET /download/{filename}
```

Downloads the generated evacuation route image.

## Project Structure

```
live_escape/
├── api/
│   └── endpoints.py          # FastAPI route handlers
├── services/
│   ├── ant_colony.py         # ACO pathfinding algorithm
│   ├── fire_model.py         # Fire spread simulation
│   ├── grid.py               # Grid representation
│   └── visualize.py          # Route visualization
├── matrix/
│   ├── matrix.csv            # Floor 0 layout
│   ├── matrix1.csv           # Floor 1 layout
│   └── matrix2.csv           # Floor 2 layout
├── output/                   # Generated visualization images
├── main.py                   # FastAPI application entry point
└── requirements.txt          # Python dependencies
```

## How It Works

### Fire Model
The fire simulation uses a diffusion-based model with three stages:
- **Initial**: Slow spread (5% diffusion rate)
- **Growth**: Moderate spread (12% diffusion rate, 1.3x amplification)
- **Spread**: Rapid spread (20% diffusion rate, 1.6x amplification)

### Pathfinding Algorithm
1. **Ant Colony Optimization**: Multiple virtual "ants" explore possible paths, depositing pheromones on successful routes
2. **A* Fallback**: If ACO doesn't find a solution, A* algorithm provides a guaranteed optimal path
3. **Safety Considerations**: Paths avoid high-intensity fire zones and maintain safety buffers
4. **Cost Function**: Balances distance, fire exposure, and turning penalties

### Navigation Instructions
The system identifies turning points and generates step-by-step instructions:
- Straight segments with distances
- Turn directions (left/right)
- Final approach to exit

## Configuration

### Fire Stage Thresholds
Adjust safety thresholds in `fire_model.py`:
```python
default_thresholds = {"initial": 0.35, "growth": 0.25, "spread": 0.20}
```

### ACO Parameters
Tune algorithm parameters in `ant_colony.py`:
```python
m_ants = 30        # Number of ants
alpha = 1.0        # Pheromone importance
beta = 5.0         # Heuristic importance
rho = 0.5          # Pheromone evaporation rate
Q = 15.0           # Pheromone deposit amount
max_iter = 50      # Maximum iterations
```

## Dependencies

- FastAPI - Web framework
- NumPy - Numerical computations
- Pandas - Data handling
- Matplotlib - Visualization
- Uvicorn - ASGI server

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
