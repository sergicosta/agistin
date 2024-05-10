def calculate_EstrgOut(E0, slope_fcr, Pinst, F):
    # Initialize EstrgOut list to store results
    EstrgOut = [0.0] * len(F)
    
    # Calculate EstrgOut for each time step
    for t in range(len(F)):
        if t == 0:
            EstrgOut[t] = E0 + (50 - F[t]) * Pinst * slope_fcr
        else:
            EstrgOut[t] = EstrgOut[t - 1] + (50 - F[t]) * Pinst * slope_fcr
    
    return EstrgOut

# Example usage:
E0 = 2000
slope_fcr = -5
Pinst = 1000
F = [50.0, 50.2, 50.2, 50.05, 50.05, 50.1, 50.2, 50.0, 50.0, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.95, 50.0, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.0, 49.8, 49.8, 49.8, 49.8]

# Calculate EstrgOut based on the example data
EstrgOut_values = calculate_EstrgOut(E0, slope_fcr, Pinst, F)

# Print the calculated values of EstrgOut for each time step
for t, value in enumerate(EstrgOut_values):
    print(f"EstrgOut[{t}] = {value}")
