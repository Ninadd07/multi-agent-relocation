# logger.py
import csv
import math

class SimulationLogger:
    def __init__(self, filename="simulation_log.csv"):
        self.filename = filename
        self.buffer = []
        # Header for the CSV file
        self.buffer.append(["Frame", "Agent_ID", "Pos_X", "Pos_Y", "Velocity", "Status", "Min_Neighbor_Dist"])
        print(f"Logger initialized. Data will be saved to {self.filename} upon exit.")

    def log_step(self, frame_count, agents):
        # We perform the logging calculation here to keep the main loop clean
        for i, agent in enumerate(agents):
            
            # Calculate distance to closest neighbor to prove "No Touching" constraint
            min_dist = 9999.0
            for other in agents:
                if other is agent: continue
                d = agent.pos.distance_to(other.pos)
                if d < min_dist:
                    min_dist = d
            
            # Format status
            status = "Moving" if agent.active else "Parked"
            if agent.waiting: status = "Waiting"

            # Create row: Frame, ID, X, Y, Speed, Status, SafetyMetric
            row = [
                frame_count,
                i,
                f"{agent.pos.x:.2f}",
                f"{agent.pos.y:.2f}",
                f"{agent.velocity.length():.2f}",
                status,
                f"{min_dist:.2f}"
            ]
            self.buffer.append(row)

    def save_log(self):
        try:
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.buffer)
            print(f"Successfully saved {len(self.buffer)} rows to {self.filename}")
        except Exception as e:
            print(f"Failed to save log: {e}")