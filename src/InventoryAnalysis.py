import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from InventorySimulation import InventorySimulation

class InventoryAnalysis:
    def __init__(self, simulations):
        """
        Initialize the analysis tool with pre-run simulations.
        
        :param simulations: List of InventorySimulation objects that have already been run
        """
        self.simulations = simulations
        self.base_params = simulations[0].__dict__.copy() if simulations else None
        
    def get_summary_table(self):
        """
        Return a DataFrame with average statistics across all simulations.
        """
        if not self.simulations:
            print("No simulations available.")
            return None
            
        results = []
        for sim in self.simulations:
            results.append({
                'Total Revenue': sim.total_revenue,
                'Total Order Cost': sim.total_order_cost,
                'Total Holding Cost': sim.total_holding_cost,
                'Avg Profit per Time': (sim.total_revenue - sim.total_order_cost - sim.total_holding_cost) / sim.simulation_time,
                'Customer Arrivals': len(sim.arrival_times)
            })
        
        df = pd.DataFrame(results)
        summary = df.mean().to_frame().T
        summary.columns = ['Ingresos Totales Promedio', 
                          'Costo de Pedidos Promedio', 
                          'Costo de Mantenimiento Promedio', 
                          'Ganancia Promedio por Unidad de Tiempo',
                          'Llegadas de Clientes Promedio']
        
        # Add base parameters for reference
        summary['Tasa de Llegadas (lambda)'] = self.base_params['customer_arrival_rate']
        summary['Tiempo de Espera'] = self.base_params['lead_time']
        summary['Punto de Reorden (s)'] = self.base_params['reorder_point']
        summary['Nivel Máximo (S)'] = self.base_params['max_inventory_level']
        
        return summary
    
    def plot_profit_distribution(self):
        """Plot histogram of average profits from simulations."""
        if not self.simulations:
            print("No simulations to analyze.")
            return
            
        profits = [(sim.total_revenue - sim.total_order_cost - sim.total_holding_cost) / sim.simulation_time 
                  for sim in self.simulations]
        
        plt.figure(figsize=(10, 6))
        plt.hist(profits, bins=30, edgecolor='black', alpha=0.7)
        plt.title('Distribución de Ganancia Promedio por Unidad de Tiempo')
        plt.xlabel('Ganancia Promedio')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.2)
        plt.show()
        
        print(f"Ganancia media: {np.mean(profits):.2f}")
        print(f"Desviación estándar: {np.std(profits):.2f}")
        print(f"Intervalo de confianza 95%: {stats.norm.interval(0.95, loc=np.mean(profits), scale=np.std(profits)/np.sqrt(len(profits)))}")
    
    def plot_customer_arrivals(self, time_window=1):
        """
        Plot histogram of customer arrivals per time window from existing simulations.
        
        :param time_window: Time window in hours for counting arrivals
        """
        if not self.simulations:
            print("No simulations to analyze.")
            return
            
        arrival_counts = []
        for sim in self.simulations:
            bins = np.arange(0, sim.simulation_time + time_window, time_window)
            counts, _ = np.histogram(sim.arrival_times, bins=bins)
            arrival_counts.extend(counts)
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(arrival_counts, bins=range(min(arrival_counts), max(arrival_counts) + 1), 
                edgecolor='black', alpha=0.7, density=True)
        
        # Compare with theoretical Poisson distribution
        mu = self.base_params['customer_arrival_rate'] * time_window
        x = np.arange(0, max(arrival_counts) + 1)
        plt.plot(x, stats.poisson.pmf(x, mu), 'r-', lw=2, label='Poisson teórico')
        
        plt.title(f'Distribución de Llegadas de Clientes cada {time_window} hora(s)')
        plt.xlabel(f'Número de llegadas por {time_window} hora(s)')
        plt.ylabel('Densidad de probabilidad')
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.show()
        
        print(f"Tasa de llegadas teórica: {mu:.2f} clientes por {time_window} hora(s)")
        print(f"Media observada: {np.mean(arrival_counts):.2f}")
        print(f"Desviación estándar observada: {np.std(arrival_counts):.2f}")
    
    def plot_cost_breakdown(self):
        """Plot average breakdown of costs and revenue."""
        if not self.simulations:
            print("No simulations to analyze.")
            return
            
        avg_revenue = np.mean([sim.total_revenue for sim in self.simulations])
        avg_order_cost = np.mean([sim.total_order_cost for sim in self.simulations])
        avg_holding_cost = np.mean([sim.total_holding_cost for sim in self.simulations])
        
        labels = ['Ingresos', 'Costos de Pedidos', 'Costos de Mantenimiento']
        values = [avg_revenue, avg_order_cost, avg_holding_cost]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color=['green', 'red', 'orange'])
        plt.title('Desglose Promedio de Costos e Ingresos')
        plt.ylabel('Monto')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.grid(True, axis='y', alpha=0.2)
        plt.show()
    
    
    def analyze_sS_sensitivity(self, s_values, S_values):
        """
        Analyze sensitivity to reorder point (s) and max inventory level (S) parameters.
        Requires simulations with varying s and S values already run and provided in constructor.
        
        :param s_values: List of s values to analyze
        :param S_values: List of S values to analyze
        :return: DataFrame with results grouped by (s, S) combinations
        """
        if not self.simulations:
            print("No simulations available for analysis.")
            return None
            
        # Agrupar simulaciones por parámetros (s, S)
        results = []
        for sim in self.simulations:
            profit = (sim.total_revenue - sim.total_order_cost - sim.total_holding_cost) / sim.simulation_time
            results.append({
                'Reorder Point (s)': sim.reorder_point,
                'Max Inventory (S)': sim.max_inventory_level,
                'Avg Profit': profit,
                'Total Revenue': sim.total_revenue,
                'Total Order Cost': sim.total_order_cost,
                'Total Holding Cost': sim.total_holding_cost,
                'Customer Arrivals': len(sim.arrival_times)
            })
        
        df = pd.DataFrame(results)
        
        # Agrupar por combinaciones (s, S)
        grouped = df.groupby(['Reorder Point (s)', 'Max Inventory (S)']).agg({
            'Avg Profit': ['mean', 'std', 'min', 'max', 'median'],
            'Total Revenue': 'mean',
            'Total Order Cost': 'mean',
            'Total Holding Cost': 'mean',
            'Customer Arrivals': 'mean'
        }).reset_index()
        
        # Renombrar columnas para mejor legibilidad
        grouped.columns = [' '.join(col).strip() for col in grouped.columns.values]
        grouped.columns = [col.replace('mean', 'Avg').replace('std', 'Std') for col in grouped.columns]
        
        return grouped
    
    def plot_sS_sensitivity(self, s_values, S_values):
        """
        Plot sensitivity analysis results for s and S parameters.
        """
        sensitivity_df = self.analyze_sS_sensitivity(s_values, S_values)
        if sensitivity_df is None:
            return
            
        # Heatmap de ganancia promedio
        pivot_table = sensitivity_df.pivot(index='Reorder Point (s)', 
                                        columns='Max Inventory (S)', 
                                        values='Avg Profit Avg')
        
        plt.figure(figsize=(12, 8))
        plt.imshow(pivot_table, cmap='viridis', aspect='auto')
        plt.colorbar(label='Ganancia Promedio')
        plt.xticks(np.arange(len(pivot_table.columns)), pivot_table.columns)
        plt.yticks(np.arange(len(pivot_table.index)), pivot_table.index)
        plt.xlabel('Nivel Máximo de Inventario (S)')
        plt.ylabel('Punto de Reorden (s)')
        plt.title('Ganancia Promedio por Combinación (s, S)')
        plt.show()
        
        # Gráfico de líneas para S fijo, variando s
        plt.figure(figsize=(12, 6))
        for S in S_values:
            subset = sensitivity_df[sensitivity_df['Max Inventory (S)'] == S]
            if not subset.empty:
                plt.plot(subset['Reorder Point (s)'], subset['Avg Profit Avg'], 
                        label=f'S={S}', marker='o')
        
        plt.xlabel('Punto de Reorden (s)')
        plt.ylabel('Ganancia Promedio')
        plt.title('Ganancia Promedio vs Punto de Reorden para diferentes Niveles Máximos')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return sensitivity_df