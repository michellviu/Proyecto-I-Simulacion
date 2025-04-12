import numpy as np
import heapq

class InventorySimulation:
  
    def __init__(self,customer_arrival_rate, holding_cost, lead_time, simulation_time, reorder_point, max_inventory_level, order_cost_function, demand_distribution_function):
        """
        Initialize the simulation parameters.
        :param selling_price: Unit selling price
        :param customer_arrival_rate: Arrival rate of customers (Poisson process)
        :param holding_cost: Inventory holding cost per unit per unit of time
        :param lead_time: Lead time for order delivery
        :param simulation_time: Total simulation time
        :param reorder_point: Reorder point (s)
        :param max_inventory_level: Order-up-to level (S)
        :param order_cost_function: Function to calculate order cost
        :param demand_distribution_function: Function to generate customer demand (distribution G)
        """
        self.selling_price = order_cost_function(1)*1.3  # Selling price is set to 30% more than the order cost of one unit
        self.customer_arrival_rate = customer_arrival_rate
        self.holding_cost = holding_cost
        self.lead_time = lead_time
        self.simulation_time = simulation_time
        self.reorder_point = reorder_point
        self.max_inventory_level = max_inventory_level
        self.order_cost_function = order_cost_function
        self.demand_distribution_function = demand_distribution_function

        # Simulation state variables
        self.current_time = 0  # Current time
        self.current_inventory = max_inventory_level  # Initial inventory level
        self.pending_order_quantity = 0  # Pending order quantity
        self.total_order_cost = 0  # Total ordering cost
        self.total_holding_cost = 0  # Total holding cost
        self.arrival_times = [] # List to store customer arrival times
        self.total_revenue = 0  # Total revenue
        self.next_order_delivery_time = float('inf')  # No pending order initially
        
        self.rng = np.random.default_rng()
        
        self.events_queue = []
        self.events = {'client_arrival': self.handle_customer_arrival,
                       'order_arrival': self.handle_order_delivery}

       
    def next_arrival(self):
        if self.current_time >= self.simulation_time:
            return
        ta = self.rng.exponential(1/self.customer_arrival_rate)
        next_ta = self.current_time + ta
        if next_ta < self.simulation_time:
            heapq.heappush(self.events_queue, (next_ta, 'client_arrival'))
    
    
    def handle_customer_arrival(self,next_ta):
        """
        Handle a customer arrival event.
        """
        # Update holding cost for the time period
        
        self.total_holding_cost += (next_ta - self.current_time) * self.current_inventory * self.holding_cost
        self.current_time = next_ta
        self.arrival_times.append(next_ta)  # Store arrival time

        # Generate customer demand (distribution G)
        customer_demand = self.demand_distribution_function()
        fulfilled_demand = min(customer_demand, self.current_inventory)
        self.total_revenue += fulfilled_demand * self.selling_price
        self.current_inventory -= fulfilled_demand

        # Check if an order is needed
        if self.current_inventory < self.reorder_point and self.pending_order_quantity == 0:
            self.pending_order_quantity = self.max_inventory_level - self.current_inventory
            self.next_order_delivery_time = self.current_time + self.lead_time
            if self.next_order_delivery_time < self.simulation_time:
                # Schedule order arrival event
                heapq.heappush(self.events_queue, (self.next_order_delivery_time, 'order_arrival'))

        # Schedule next customer arrival
        self.next_arrival()

    def handle_order_delivery(self,next_ta):
        """
        Handle an order delivery event.
        """
        # Update holding cost for the time period
        self.total_holding_cost += (next_ta - self.current_time) * self.current_inventory * self.holding_cost
        self.current_time = next_ta

        # Update inventory and costs
        self.total_order_cost += self.order_cost_function(self.pending_order_quantity)
        self.current_inventory += self.pending_order_quantity
        self.pending_order_quantity = 0
        self.next_order_delivery_time = float('inf')
    
    
    def run(self):
        """
        Run the simulation until the specified simulation time.
        """
        self.next_arrival()
        while self.events_queue:
            event_tuple = heapq.heappop(self.events_queue)
            time,event_type = event_tuple
            if event_type == 'client_arrival':
                self.events[event_type](time)
            elif event_type == 'order_arrival':
                self.events[event_type](time)
        # Calculate average profit per unit time
        average_profit = (self.total_revenue - self.total_order_cost - self.total_holding_cost) / self.simulation_time
        return average_profit

    def summary(self):
        """
        Return a summary of the simulation results.
        """
        return {
            "Total Revenue": self.total_revenue,
            "Total Ordering Cost": self.total_order_cost,
            "Total Holding Cost": self.total_holding_cost,
            "Average Profit per Unit Time": (self.total_revenue - self.total_order_cost - self.total_holding_cost) / self.simulation_time
        }


# Example usage
if __name__ == "__main__":
    # Define cost function and demand distribution
    def order_cost_function(order_quantity):
        return 5 + 5 * order_quantity  # Fixed cost + variable cost per unit

    def demand_distribution_function():
        return np.random.geometric(p=0.3)  # Example of distribution G (geometric)

    # Initialize and run simulation
    simulation = InventorySimulation(
        customer_arrival_rate=5,  # Poisson arrival rate (lambda)
        holding_cost=1,
        lead_time=1,
        simulation_time=24,
        reorder_point=20,
        max_inventory_level=80,
        order_cost_function=order_cost_function,
        demand_distribution_function=demand_distribution_function
    )
    average_profit = simulation.run()
    print(f"Average Profit per Unit Time: {average_profit:.2f}")
    print(simulation.summary())