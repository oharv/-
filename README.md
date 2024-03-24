# -
スマートグリッドは、ブロックチェーン技術と機械学習を活用して、再生可能エネルギーの生産と分配を最適化し、エネルギーの安定供給と効率的な利用を実現します。
import hashlib
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Basic Blockchain Implementation
class Block:
    def __init__(self, index, timestamp, data, previous_hash=' '):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = str(self.index) + str(self.timestamp) + str(self.data) + self.previous_hash
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, datetime.datetime.now(), "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]

            if current_block.hash != current_block.calculate_hash():
                return False

            if current_block.previous_hash != previous_block.hash:
                return False
        return True

# Machine Learning for Energy Demand Prediction
class EnergyDemandPredictor:
    def __init__(self, data):
        self.data = data

    def train_model(self):
        # Assuming 'data' is a 2D array where rows are [hour_of_day, energy_demand]
        X = self.data[:, 0].reshape(-1, 1)  # Features (hour of day)
        y = self.data[:, 1]  # Target (energy demand)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Model trained with MSE: {mse}")

    def predict(self, hour_of_day):
        return self.model.predict(np.array([[hour_of_day]]))[0]

# Example Usage
if __name__ == "__main__":
    # Blockchain
    my_blockchain = Blockchain()
    my_blockchain.add_block(Block(1, datetime.datetime.now(), "Block 1 Data"))
    print("Blockchain valid?", my_blockchain.is_chain_valid())

    # Machine Learning
    # Example data: hour of day (0-23) vs energy demand
    energy_data = np.array([
        [0, 300],
        [6, 450],
        [12, 800],
        [18, 670],
        [23, 300]
    ])
    predictor = EnergyDemandPredictor(energy_data)
    predictor.train_model()
    print("Predicted energy demand at 15:00:", predictor.predict(15))
