# AI & Machine Learning Course Projects

This repository contains implementations for two distinct Artificial Intelligence assignments: a high-performance **Game AI agent for Reversed Reversi** and a **Data Mining pipeline for Census Income classification**.

---

## 📂 Project 1: Reversed Reversi AI Agent

### **Overview**
This project implements an AI agent to play **Reversed Reversi** (also known as Anti-Reversi). Unlike standard Reversi, the goal is to have the *fewest* pieces on the board when the game ends.

### **Key Features & Strategy**
The agent is built using a **Minimax algorithm with Alpha-Beta Pruning** and includes several optimizations to meet strict time (5s/move) and memory limits:

#### **Numba Optimization:**
- Utilizes `@njit` from the `numba` library to compile the move generation and flipping logic into machine code, significantly speeding up the search tree exploration.

#### **Iterative Deepening:**
- The agent searches deeper incrementally (Depth 1, 2, ... N) to ensure it always has a valid move ready if the time limit approaches.

#### **Dynamic Heuristic Evaluation:**
The evaluation function adapts weights based on the game stage (Opening, Mid-game, Endgame):
- **Mobility:** Prioritizes having more legal moves than the opponent (mobility) and "quiet moves" (moves that don't flip many discs).
- **Corners & Stability:** Heavily rewards capturing corners and "stable edges" that cannot be re-flipped.
- **Disc Parity:** In the endgame (empty squares < 10), the logic switches to aggressively minimizing disc count.

### **Tech Stack**
- **Python 3.7+**
- **NumPy:** For efficient board representation and vector operations.
- **Numba:** For JIT compilation of computationally expensive loops.

### **Usage**
The agent is encapsulated in the `AI` class.

```python
import numpy as np
from ai_agent import AI  # Assuming script is named ai_agent.py

# Initialize Agent (Size 8x8, Color Black=-1, Timeout 5s)
agent = AI(chessboard_size=8, color=-1, time_out=5)

# Get best move
current_board = np.zeros((8,8)) # Replace with actual board state
agent.go(current_board)
best_move = agent.candidate_list[-1]
```

---

## 📂 Project 2: Adult Census Income Prediction

### **Overview**
This project builds a robust machine learning pipeline to predict whether an individual earns more than **$50K/year** based on 14 demographic and employment attributes.

### **Methodology**
The solution is implemented using `scikit-learn` and follows a rigorous data mining workflow:

#### **Data Preprocessing Pipeline:**
- **Imputation:** Fills missing values using 'median' for numerical data and 'most_frequent' for categorical data.
- **Encoding:** One-Hot Encoding for categorical features (e.g., workclass, education, marital-status).
- **Scaling:** Standard scaling for numerical features (e.g., age, capital-gain).

#### **Model Benchmarking:**
The script automatically performs a **Grid Search with Cross-Validation (4-fold)** across a variety of algorithms to find the champion model:
- Logistic Regression
- Decision Trees & Random Forests
- Support Vector Machines (SVM - RBF Kernel)
- K-Nearest Neighbors (KNN)
- Neural Networks (MLP Classifier)
- Gradient Boosting (Classic & Histogram-based)

#### **Evaluation:**
- Generates a confusion matrix heatmap.
- Outputs a classification report (Precision, Recall, F1-Score).
- Retrains the best-performing model on the full dataset for final predictions.

### **Tech Stack**
- **Pandas:** Data manipulation.
- **Scikit-Learn:** Modeling, Pipelines, and GridSearch.
- **Seaborn/Matplotlib:** Visualization of results.

### **Usage**
Ensure `traindata.csv`, `testdata.csv`, and `trainlabel.txt` are in the directory.

```bash
python train_census_model.py
```

The script will output the best model's accuracy, save a confusion matrix image, and generate `testlabel.txt` with predictions.

---

## 🛠️ Installation Requirements

To run both projects, install the dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn numba
```

---

## 📜 Credits

**Assignments:** CS303 Data Mining & AI Algorithms Courses.  
**Census Dataset:** Ronny Kohavi and Barry Becker (1994 Census Bureau database).
