from InventoryAnalysis import InventoryAnalysis
from InventorySimulation import InventorySimulation
import numpy as np
# Example usage
if __name__ == "__main__":
    # Define base parameters
    def order_cost_function(order_quantity):
            return 5 + 5 * order_quantity  # Fixed cost + variable cost per unit

    def demand_distribution_function():
            return np.random.geometric(p=0.3)  # Example of distribution G (geometric)

    # Configuración de parámetros base
    base_params = {
        'customer_arrival_rate': 5,
        'holding_cost': 1,
        'lead_time': 1,
        'simulation_time': 24,
        'order_cost_function': order_cost_function,
        'demand_distribution_function': demand_distribution_function
    }

    simulations = []
    for _ in range(1000):  # 100 simulaciones
        sim = InventorySimulation(
            customer_arrival_rate=5,
            holding_cost=1,
            lead_time=1,
            simulation_time=24,
            reorder_point=20,
            max_inventory_level=80,
            order_cost_function=order_cost_function,
            demand_distribution_function=demand_distribution_function
        )
        sim.run()
        simulations.append(sim)

    # 2. Crear el analizador con las simulaciones ya ejecutadas
    analyzer = InventoryAnalysis(simulations)

    # 3. Generar todos los análisis y gráficos
    print("\nTabla de Resumen de Promedios:")
    print(analyzer.get_summary_table().to_string(index=False))

    analyzer.plot_profit_distribution()
    analyzer.plot_customer_arrivals(time_window=1)  # Llegadas por hora
    analyzer.plot_cost_breakdown()

    # Valores a probar para s y S
    s_values = [10, 15, 20, 25, 30]
    S_values = [50, 60, 70, 80, 90]

    # Ejecutar simulaciones para todas las combinaciones
    simulations = []
    for s in s_values:
        for S in S_values:
            if S > s:  # S debe ser mayor que s
                for _ in range(20):  # 20 simulaciones por combinación
                    params = base_params.copy()
                    params['reorder_point'] = s
                    params['max_inventory_level'] = S
                    
                    sim = InventorySimulation(**params)
                    sim.run()
                    simulations.append(sim)
    
    # Crear el analizador con todas las simulaciones
    analyzer = InventoryAnalysis(simulations)

    # Realizar y mostrar análisis de sensibilidad
    sensitivity_results = analyzer.plot_sS_sensitivity(s_values, S_values)

    # Mostrar tabla con los mejores parámetros
    print("\nMejores combinaciones de parámetros:")
    print(sensitivity_results.sort_values('Avg Profit Avg', ascending=False).head(10).to_string(index=False))