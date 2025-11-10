from vehicle_model import VehicleModel
import numpy as np
import matplotlib.pyplot as plt
from ex05_Tuning_PID_Controller import PID_Controller

if __name__ == "__main__":
    target_y = 0.0
    step_time = 0.1
    simulation_time = 30   

    PID_sets = [
        {"P": 1.0, "D": 1.0, "I": 0.1, "label": "P=1.0, D=1.0, I=0.1"},
        {"P": 2.0, "D": 3.0, "I": 0.4, "label": "P=2.0, D=3.0, I=0.4 "},
        {"P": 3.0, "D": 2.0, "I": 0.0, "label": "P=3.0, D=2.0, I=0.0"},
        {"P": 0.8, "D": 5.0, "I": 0.2, "label": "P=0.8, D=5.0, I=0.2 "}
    ]

    plt.figure(figsize=(10, 6))

    for pid in PID_sets:
        plant = VehicleModel(step_time, 0.0, 0.4, -0.1)
        controller = PID_Controller(target_y, plant.y_measure[0][0], step_time,
                                    P_Gain=pid["P"], D_Gain=pid["D"], I_Gain=pid["I"])
        
        measure_y = []
        time = []

        for i in range(int(simulation_time/step_time)):
            time.append(step_time*i)
            measure_y.append(plant.y_measure[0][0])
            controller.ControllerInput(target_y, plant.y_measure[0][0])
            plant.ControlInput(controller.u)
        
        plt.plot(time, measure_y, label=pid["label"])

    plt.plot([0, simulation_time], [target_y, target_y], 'k--', linewidth=1.0, label="Reference")

    plt.xlabel('Time [s]')
    plt.ylabel('Vehicle Position')
    plt.title('PID Gain Comparison on Vehicle Model')
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()